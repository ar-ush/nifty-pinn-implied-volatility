"""
================================================================
get_iv.py — Single Option Implied Volatility Calculator
================================================================
For trading use. Input any Nifty option details and get:
  1. Classical IV  — Newton-Raphson, instant, exact
  2. PINN IV       — Physics-informed, slower, regularized
  3. Surface IV    — From trained vol surface (if available)
  4. RV Forecast   — Predicted next month realized vol

USAGE:
  # From command line
  python get_iv.py --S 24500 --K 24500 --r 0.065 --T 14 --C 185.50

  # From Python
  from get_iv import get_iv
  result = get_iv(S=24500, K=24500, r=0.065, T=14, C=185.50)
  print(result)

ARGUMENTS:
  S  = Nifty spot price right now
  K  = Strike price of the option
  r  = Risk-free rate (annualized decimal, e.g. 0.065 = 6.5%)
  T  = Days to expiry
  C  = Option market price (last traded price or mid of bid-ask)
  type = 'call' or 'put' (default: call)

================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
# SECTION 1: CLASSICAL IV (Newton-Raphson) — instant
# ================================================================

def classical_iv(C, S, K, r, T, option_type='call'):
    """
    Newton-Raphson + Brent fallback.
    Returns IV in decimal (0.12 = 12%).
    Runs in < 1 millisecond.
    """
    T_yr = T / 252  # convert days to years

    if T_yr <= 0 or C <= 0:
        return np.nan

    # No-arbitrage check
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T_yr), 0)
    else:
        intrinsic = max(K * np.exp(-r * T_yr) - S, 0)

    if C < intrinsic - 0.01:
        return np.nan

    def bs_price(sigma):
        if sigma <= 0:
            return intrinsic
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
        d2 = d1 - sigma*np.sqrt(T_yr)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T_yr)*norm.cdf(d2)
        return K*np.exp(-r*T_yr)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def bs_vega(sigma):
        if sigma <= 0:
            return 1e-10
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
        return S * norm.pdf(d1) * np.sqrt(T_yr)

    # Brenner-Subrahmanyam initial guess
    sigma = np.sqrt(2*np.pi/T_yr) * C / S
    sigma = np.clip(sigma, 0.01, 5.0)

    # Newton-Raphson
    for _ in range(200):
        price = bs_price(sigma)
        vega  = bs_vega(sigma)
        diff  = price - C
        if abs(diff) < 1e-8:
            return float(sigma)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma  = max(sigma, 1e-6)

    # Brent fallback
    try:
        return float(brentq(
            lambda s: bs_price(s) - C,
            1e-6, 10.0, xtol=1e-8))
    except Exception:
        return np.nan


# ================================================================
# SECTION 2: PINN ARCHITECTURE
# ================================================================

class BSPINN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=5):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        list(self.net.modules())[-1].bias.data.fill_(-4.0)

    def forward(self, tau, s):
        x = torch.cat([tau, s], dim=1)
        return torch.nn.functional.softplus(self.net(x))


def bs_call_torch(S, K, r, T, sigma):
    eps  = 1e-8
    d1   = (torch.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*torch.sqrt(T)+eps)
    d2   = d1 - sigma*torch.sqrt(T)
    n_d1 = 0.5*(1.0 + torch.erf(d1/np.sqrt(2)))
    n_d2 = 0.5*(1.0 + torch.erf(d2/np.sqrt(2)))
    return S*n_d1 - K*torch.exp(-r*T)*n_d2


def pde_residual(net, tau, s, sigma, r):
    c    = net(tau, s)
    ones = torch.ones_like(c)
    dc_dtau = torch.autograd.grad(
        c, tau, grad_outputs=ones,
        create_graph=True, retain_graph=True)[0]
    dc_ds   = torch.autograd.grad(
        c, s, grad_outputs=ones,
        create_graph=True, retain_graph=True)[0]
    d2c_ds2 = torch.autograd.grad(
        dc_ds, s, grad_outputs=torch.ones_like(dc_ds),
        create_graph=True, retain_graph=True)[0]
    return (-dc_dtau + 0.5*sigma**2*s**2*d2c_ds2
            + r*s*dc_ds - r*c)


def bc_loss(net, tau_min, tau_max, r,
            s_max=1.2, n=300, device=DEVICE):
    loss = torch.tensor(0.0, device=device)
    s1   = torch.rand(n,1,device=device)*(s_max-0.5)+0.5
    tau1 = torch.zeros(n,1,device=device)
    loss = loss + torch.mean((net(tau1,s1)-torch.relu(s1-1.0))**2)
    tau2 = torch.rand(n,1,device=device)*(tau_max-tau_min)+tau_min
    s2   = torch.full((n,1),0.5,device=device)
    loss = loss + torch.mean(net(tau2,s2)**2)
    tau3 = torch.rand(n,1,device=device)*(tau_max-tau_min)+tau_min
    s3   = torch.full((n,1),s_max,device=device)
    loss = loss + torch.mean((net(tau3,s3)-(s3-torch.exp(-r*tau3)))**2)
    return loss


# ================================================================
# SECTION 3: PINN IV FOR A SINGLE OPTION
# ================================================================

def pinn_iv_single(C, S, K, r, T,
                   sigma_init=None,
                   n_epochs=5000,
                   verbose=False):
    """
    Extracts PINN IV for a single option.

    The PINN treats this single observation as a 1-point window.
    For better results, pass a small dataframe of recent options
    via pinn_iv_window() below — more data = better PDE constraint.

    C, S, K : prices in rupees
    r       : annualized rate (decimal)
    T       : days to expiry
    """
    T_yr = T / 252

    if sigma_init is None:
        # Use classical IV as starting point
        sigma_init = classical_iv(C, S, K, r, T)
        if sigma_init is None or np.isnan(sigma_init):
            sigma_init = 0.15

    # Build a mini synthetic window around this point
    # by slightly perturbing moneyness — gives PINN
    # enough variation to learn the pricing surface
    moneyness = S / K
    n_pts     = 10

    # Small perturbations around actual moneyness
    mono_pts = np.linspace(
        max(moneyness - 0.03, 0.85),
        min(moneyness + 0.03, 1.15),
        n_pts)

    # Generate synthetic prices for nearby strikes
    # using classical IV as the vol estimate
    syn_S = np.full(n_pts, S)
    syn_K = S / mono_pts
    syn_T = np.full(n_pts, T_yr)
    syn_C = []
    for mk in syn_K:
        p = _bs_price(S, mk, r, T_yr, sigma_init)
        syn_C.append(p if p > 0.5 else 0.5)

    # Replace center point with actual market price
    center = n_pts // 2
    syn_C[center] = C
    syn_K[center] = K

    df_mini = pd.DataFrame({
        'Spot'      : syn_S,
        'Strike'    : syn_K,
        'C_market'  : np.array(syn_C),
        'T_years'   : syn_T,
        'Moneyness' : syn_S / syn_K,
        'r'         : np.full(n_pts, r),
    })

    # Train inverse PINN
    r_t   = torch.tensor(float(r), device=DEVICE)
    net   = BSPINN().to(DEVICE)
    log_s = nn.Parameter(torch.tensor(
        float(np.log(max(sigma_init, 0.01))),
        dtype=torch.float32, device=DEVICE))
    log_s_init = float(np.log(max(sigma_init, 0.01)))

    optimizer = optim.Adam([
        {'params': net.parameters(), 'lr': 1e-3},
        {'params': [log_s],          'lr': 1e-2},
    ])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)

    tau_np = df_mini['T_years'].values.astype(np.float32)
    s_np   = df_mini['Moneyness'].values.astype(np.float32)
    c_np   = (df_mini['C_market']/df_mini['Strike']).values.astype(np.float32)
    C_np   = df_mini['C_market'].values.astype(np.float32)
    S_np   = df_mini['Spot'].values.astype(np.float32)
    K_np   = df_mini['Strike'].values.astype(np.float32)

    tau_d = torch.tensor(tau_np, device=DEVICE).unsqueeze(1)
    s_d   = torch.tensor(s_np,   device=DEVICE).unsqueeze(1)
    c_d   = torch.tensor(c_np,   device=DEVICE).unsqueeze(1)
    C_d   = torch.tensor(C_np,   device=DEVICE).unsqueeze(1)
    S_d   = torch.tensor(S_np,   device=DEVICE).unsqueeze(1)
    K_d   = torch.tensor(K_np,   device=DEVICE).unsqueeze(1)
    T_d   = tau_d.clone()

    tau_min = float(tau_np.min())*0.5
    tau_max = float(tau_np.max())*1.5
    s_lo, s_hi = 0.85, 1.15

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        sigma = torch.exp(log_s)

        tau_f = (torch.rand(1000,1,device=DEVICE,requires_grad=True)
                 *(tau_max-tau_min)+tau_min)
        s_f   = (torch.rand(1000,1,device=DEVICE,requires_grad=True)
                 *(s_hi-s_lo)+s_lo)

        loss_pde   = torch.mean(pde_residual(net,tau_f,s_f,sigma,r_t)**2)
        loss_bc    = bc_loss(net,tau_min,tau_max,r_t,device=DEVICE)
        loss_data  = torch.mean((net(tau_d,s_d)-c_d)**2)
        C_bs       = bs_call_torch(S_d,K_d,r_t,T_d,sigma)
        loss_iv    = torch.mean((C_bs-C_d)**2)
        loss_prior = (log_s - log_s_init)**2

        loss = (1.0*loss_pde + 10.0*loss_bc +
                50.0*loss_data + 100.0*loss_iv +
                0.1*loss_prior)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch+1) % 1000 == 0:
            print(f"    Epoch {epoch+1} | "
                  f"sigma={torch.exp(log_s).item()*100:.2f}%")

    return float(torch.exp(log_s).item())


def _bs_price(S, K, r, T, sigma, opt='call'):
    """Helper BS pricer in pure numpy."""
    if T <= 0 or sigma <= 0:
        return max(S-K, 0) if opt=='call' else max(K-S, 0)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 'call':
        return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))


# ================================================================
# SECTION 4: FORECAST NEXT MONTH RV
# ================================================================

def forecast_next_rv(pinn_iv,
                     history_file='nifty_pinn_iv_results.csv',
                     classical_file='nifty_atm_iv_classical.csv'):
    """
    Given current PINN IV, forecast next month realized vol.
    Returns point forecast and 95% confidence interval.
    """
    import statsmodels.api as sm

    if not os.path.exists(history_file):
        return None

    df_p = pd.read_csv(history_file)
    df_p['Window_Start'] = pd.to_datetime(df_p['Window_Start'])
    df_p['Window_End']   = pd.to_datetime(df_p['Window_End'])

    df_c = pd.read_csv(classical_file)
    df_c['Date'] = pd.to_datetime(df_c['Date'])

    records = []
    for _, row in df_p.iterrows():
        mask = ((df_c['Date'] >= row['Window_Start']) &
                (df_c['Date'] <= row['Window_End']))
        win  = df_c[mask]
        if len(win) == 0:
            continue
        records.append({
            'PINN_IV'   : row['PINN_IV'],
            'RV_current': win['RV_22'].mean(),
        })

    df = pd.DataFrame(records)
    df['RV_next'] = df['RV_current'].shift(-1)
    df = df.dropna()

    if len(df) < 4:
        return None

    y = df['RV_next']
    X = sm.add_constant(df[['PINN_IV']])
    m = sm.OLS(y, X).fit()

    X_new = pd.DataFrame({'const': [1.0], 'PINN_IV': [pinn_iv]})
    pred  = m.get_prediction(X_new).summary_frame(alpha=0.05)

    return {
        'point'    : float(pred['mean'].values[0]),
        'ci_lower' : float(max(pred['obs_ci_lower'].values[0], 0)),
        'ci_upper' : float(pred['obs_ci_upper'].values[0]),
        'r2'       : float(m.rsquared),
        'p_beta'   : float(m.pvalues['PINN_IV']),
    }


# ================================================================
# SECTION 5: MAIN get_iv FUNCTION
# ================================================================

def get_iv(S, K, r, T, C, option_type='call',
           use_pinn=True, verbose=False):
    """
    ================================================================
    MAIN FUNCTION — Call this for any option

    Parameters:
      S           : Nifty spot price (e.g. 24500)
      K           : Strike price     (e.g. 24500)
      r           : Risk-free rate   (e.g. 0.065 for 6.5%)
      T           : Days to expiry   (e.g. 14)
      C           : Option price     (e.g. 185.50)
      option_type : 'call' or 'put'  (default: 'call')
      use_pinn    : True = also compute PINN IV (slower)
                    False = classical only (instant)
      verbose     : True = print PINN training progress

    Returns dict with:
      IV_classical : Newton-Raphson IV (instant)
      IV_pinn      : Physics-informed IV (if use_pinn=True)
      RV_forecast  : Predicted next month RV
      CI_lower     : 95% CI lower bound
      CI_upper     : 95% CI upper bound
      moneyness    : S/K
      T_years      : T/252
      BS_price_check: BS price using extracted IV (validation)
    ================================================================
    """
    T_yr      = T / 252
    moneyness = S / K

    print(f"\n{'='*55}")
    print(f"  OPTION IV CALCULATOR")
    print(f"{'='*55}")
    print(f"  Spot (S)       : {S:,.2f}")
    print(f"  Strike (K)     : {K:,.2f}")
    print(f"  Moneyness(S/K) : {moneyness:.4f}  "
          f"({'ATM' if abs(moneyness-1)<0.02 else 'ITM' if moneyness>1 else 'OTM'})")
    print(f"  Days to expiry : {T}")
    print(f"  Risk-free rate : {r*100:.2f}%")
    print(f"  Market price   : Rs {C:.2f}")
    print(f"  Option type    : {option_type.upper()}")
    print(f"{'='*55}")

    result = {
        'S'           : S,
        'K'           : K,
        'r'           : r,
        'T_days'      : T,
        'T_years'     : T_yr,
        'C_market'    : C,
        'option_type' : option_type,
        'moneyness'   : moneyness,
        'IV_classical': None,
        'IV_pinn'     : None,
        'RV_forecast' : None,
        'CI_lower'    : None,
        'CI_upper'    : None,
        'BS_price_check': None,
    }

    # ── Step 1: Classical IV (Newton-Raphson) ─────────────────
    print(f"\n  [1/3] Classical IV (Newton-Raphson)...")
    iv_cl = classical_iv(C, S, K, r, T, option_type)

    if iv_cl is None or np.isnan(iv_cl):
        print(f"  Classical IV   : FAILED (check inputs)")
        print(f"  Hint: Is the price above intrinsic value?")
        intrinsic = max(S-K*np.exp(-r*T_yr),0) if option_type=='call' \
                    else max(K*np.exp(-r*T_yr)-S,0)
        print(f"  Intrinsic value: Rs {intrinsic:.2f}")
        print(f"  Market price   : Rs {C:.2f}")
        return result

    result['IV_classical'] = iv_cl

    # Validate: reprice using extracted IV
    bs_check = _bs_price(S, K, r, T_yr, iv_cl, option_type)
    result['BS_price_check'] = bs_check

    print(f"  Classical IV   : {iv_cl*100:.4f}%")
    print(f"  BS reprice     : Rs {bs_check:.4f}  "
          f"(error: Rs {abs(bs_check-C):.6f})")

    # ── Step 2: PINN IV ────────────────────────────────────────
    if use_pinn:
        print(f"\n  [2/3] PINN IV (physics-informed)...")
        print(f"  Training ~5000 epochs — takes ~2 minutes on CPU")
        print(f"  (set use_pinn=False for instant classical only)")

        iv_pinn = pinn_iv_single(
            C, S, K, r, T,
            sigma_init=iv_cl,
            n_epochs=5000,
            verbose=verbose)

        result['IV_pinn'] = iv_pinn
        print(f"  PINN IV        : {iv_pinn*100:.4f}%")
        print(f"  Difference     : "
              f"{abs(iv_pinn-iv_cl)*100:.4f}pp vs Classical")
    else:
        print(f"\n  [2/3] PINN IV skipped (use_pinn=False)")

    # ── Step 3: Forecast next month RV ────────────────────────
    print(f"\n  [3/3] Forecasting next month realized vol...")
    iv_for_forecast = result['IV_pinn'] if result['IV_pinn'] \
                      else result['IV_classical']
    forecast = forecast_next_rv(iv_for_forecast)

    if forecast:
        result['RV_forecast'] = forecast['point']
        result['CI_lower']    = forecast['ci_lower']
        result['CI_upper']    = forecast['ci_upper']
        result['forecast_r2'] = forecast['r2']
        print(f"  Predicted RV   : {forecast['point']*100:.2f}%")
        print(f"  95% CI         : [{forecast['ci_lower']*100:.2f}%, "
              f"{forecast['ci_upper']*100:.2f}%]")
        print(f"  Model R²       : {forecast['r2']:.4f}")
    else:
        print(f"  Forecast unavailable — run step2 first to "
              f"build historical IV series")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Classical IV   : {iv_cl*100:.2f}%")
    if result['IV_pinn']:
        print(f"  PINN IV        : {result['IV_pinn']*100:.2f}%")
    if result['RV_forecast']:
        print(f"  Forecast RV    : {result['RV_forecast']*100:.2f}%"
              f"  (next month)")
        print(f"  95% CI         : "
              f"[{result['CI_lower']*100:.2f}%, "
              f"{result['CI_upper']*100:.2f}%]")
    print(f"\n  TRADING SIGNALS:")
    if result['IV_classical']:
        if result['RV_forecast']:
            if result['IV_classical'] < result['RV_forecast']:
                print(f"  >> IV ({iv_cl*100:.1f}%) < Forecast RV "
                      f"({result['RV_forecast']*100:.1f}%)")
                print(f"     Options may be UNDERPRICED relative "
                      f"to expected vol")
                print(f"     Consider: Long volatility strategies "
                      f"(buy straddle/strangle)")
            else:
                print(f"  >> IV ({iv_cl*100:.1f}%) > Forecast RV "
                      f"({result['RV_forecast']*100:.1f}%)")
                print(f"     Options may be OVERPRICED relative "
                      f"to expected vol")
                print(f"     Consider: Short volatility strategies "
                      f"(sell covered calls/cash secured puts)")
    print(f"{'='*55}")
    print(f"  NOTE: This is a research model. Always validate")
    print(f"  signals with your own risk management framework.")
    print(f"{'='*55}\n")

    return result


# ================================================================
# COMMAND LINE INTERFACE
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get Implied Volatility for any Nifty option')
    parser.add_argument('--S',    type=float, required=True,
                        help='Spot price (e.g. 24500)')
    parser.add_argument('--K',    type=float, required=True,
                        help='Strike price (e.g. 24500)')
    parser.add_argument('--r',    type=float, default=0.065,
                        help='Risk-free rate (e.g. 0.065)')
    parser.add_argument('--T',    type=int,   required=True,
                        help='Days to expiry (e.g. 14)')
    parser.add_argument('--C',    type=float, required=True,
                        help='Option market price (e.g. 185.50)')
    parser.add_argument('--type', type=str,   default='call',
                        choices=['call','put'],
                        help='Option type (default: call)')
    parser.add_argument('--no-pinn', action='store_true',
                        help='Skip PINN, use classical only (faster)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print PINN training progress')
    args = parser.parse_args()

    result = get_iv(
        S           = args.S,
        K           = args.K,
        r           = args.r,
        T           = args.T,
        C           = args.C,
        option_type = args.type,
        use_pinn    = not args.no_pinn,
        verbose     = args.verbose,
    )