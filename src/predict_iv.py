"""
================================================================
IV PREDICTION PIPELINE
Two Modes:

MODE A — Real-time IV Extraction
  Input : Recent NSE option prices (last 22 days)
  Output: Today's implied volatility via Inverse PINN

MODE B — Future RV Forecasting
  Input : Extracted PINN IV
  Output: Predicted realized volatility next month

USAGE:
  # Mode A only
  python predict_iv.py --mode extract

  # Mode B only (requires nifty_pinn_iv_results.csv)
  python predict_iv.py --mode forecast

  # Both together (recommended)
  python predict_iv.py --mode both

REQUIREMENTS:
  nifty_atm_iv_classical.csv       (from step1)
  nifty_pinn_iv_results.csv        (from step2)
  step4_regression_results needed  (fitted internally here)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
# SECTION 1: PINN ARCHITECTURE (same as step2)
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
    d1   = (torch.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*torch.sqrt(T) + eps)
    d2   = d1 - sigma*torch.sqrt(T)
    n_d1 = 0.5*(1.0 + torch.erf(d1/np.sqrt(2)))
    n_d2 = 0.5*(1.0 + torch.erf(d2/np.sqrt(2)))
    return S*n_d1 - K*torch.exp(-r*T)*n_d2


def bs_pde_residual(net, tau, s, sigma, r):
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
    return (-dc_dtau
            + 0.5*sigma**2*s**2*d2c_ds2
            + r*s*dc_ds - r*c)


def boundary_loss(net, tau_min, tau_max, r,
                  s_max=1.2, n_bc=300, device=DEVICE):
    loss = torch.tensor(0.0, device=device)
    s1   = torch.rand(n_bc, 1, device=device)*(s_max-0.5)+0.5
    tau1 = torch.zeros(n_bc, 1, device=device)
    loss = loss + torch.mean((net(tau1,s1) - torch.relu(s1-1.0))**2)
    tau2 = torch.rand(n_bc, 1, device=device)*(tau_max-tau_min)+tau_min
    s2   = torch.full((n_bc,1), 0.5, device=device)
    loss = loss + torch.mean(net(tau2,s2)**2)
    tau3 = torch.rand(n_bc, 1, device=device)*(tau_max-tau_min)+tau_min
    s3   = torch.full((n_bc,1), s_max, device=device)
    loss = loss + torch.mean((net(tau3,s3) - (s3-torch.exp(-r*tau3)))**2)
    return loss


# ================================================================
# SECTION 2: INVERSE PINN (same as step2 v4)
# ================================================================

class InversePINN:
    def __init__(self, sigma_init, r, device=DEVICE):
        self.r      = torch.tensor(float(r), device=device)
        self.device = device
        self.net    = BSPINN().to(device)
        self.log_sigma = nn.Parameter(
            torch.tensor(float(np.log(max(sigma_init, 0.01))),
                         dtype=torch.float32, device=device))
        self.log_sigma_init = float(np.log(max(sigma_init, 0.01)))
        self.sigma_trace = []

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def get_pinn_iv(self):
        return self.sigma.item()

    def train(self, df_window, n_epochs=5000,
              lr_net=1e-3, lr_sigma=1e-2,
              n_colloc=2000,
              lambda_pde=1.0, lambda_bc=10.0,
              lambda_data=50.0, lambda_iv=100.0,
              lambda_prior=0.1, verbose=True):

        optimizer = optim.Adam([
            {'params': self.net.parameters(), 'lr': lr_net},
            {'params': [self.log_sigma],       'lr': lr_sigma},
        ])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=2000, gamma=0.5)

        tau_np = df_window['T_years'].values.astype(np.float32)
        s_np   = df_window['Moneyness'].values.astype(np.float32)
        c_np   = (df_window['C_market']/df_window['Strike']).values.astype(np.float32)
        C_np   = df_window['C_market'].values.astype(np.float32)
        S_np   = df_window['Spot'].values.astype(np.float32)
        K_np   = df_window['Strike'].values.astype(np.float32)

        tau_data = torch.tensor(tau_np, device=self.device).unsqueeze(1)
        s_data   = torch.tensor(s_np,   device=self.device).unsqueeze(1)
        c_data   = torch.tensor(c_np,   device=self.device).unsqueeze(1)
        C_data   = torch.tensor(C_np,   device=self.device).unsqueeze(1)
        S_data   = torch.tensor(S_np,   device=self.device).unsqueeze(1)
        K_data   = torch.tensor(K_np,   device=self.device).unsqueeze(1)
        T_data   = tau_data.clone()

        tau_min = float(tau_np.min())*0.5
        tau_max = float(tau_np.max())*1.5
        s_lo, s_hi = 0.85, 1.15
        log_rv = self.log_sigma_init

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            tau_f = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True)*(tau_max-tau_min)+tau_min)
            s_f   = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True)*(s_hi-s_lo)+s_lo)

            loss_pde   = torch.mean(bs_pde_residual(
                self.net, tau_f, s_f, self.sigma, self.r)**2)
            loss_bc    = boundary_loss(
                self.net, tau_min, tau_max, self.r, device=self.device)
            loss_data  = torch.mean(
                (self.net(tau_data, s_data)-c_data)**2)
            C_pred_bs  = bs_call_torch(
                S_data, K_data, self.r, T_data, self.sigma)
            loss_iv    = torch.mean((C_pred_bs-C_data)**2)
            loss_prior = (self.log_sigma-log_rv)**2

            loss = (lambda_pde   * loss_pde
                  + lambda_bc    * loss_bc
                  + lambda_data  * loss_data
                  + lambda_iv    * loss_iv
                  + lambda_prior * loss_prior)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            self.sigma_trace.append(self.sigma.item())

            if verbose and (epoch+1) % 1000 == 0:
                print(f"    Epoch {epoch+1:4d} | "
                      f"sigma={self.sigma.item()*100:.2f}% | "
                      f"IV_loss={loss_iv.item():.3e}")

        return self


# ================================================================
# SECTION 3: MODE A — REAL-TIME IV EXTRACTION
# ================================================================

def extract_iv_today(df_window, verbose=True):
    """
    Given a 22-day window of ATM option prices,
    extracts today's implied volatility via Inverse PINN.

    This is the operational real-time use of the model.
    Call this whenever you have fresh NSE data.

    Input dataframe needs columns:
      Date, Spot, Strike, C_market, T_years, Moneyness, r, RV_22

    Returns:
      pinn_iv    : float — extracted implied volatility
      classical_iv: float — Newton-Raphson IV for comparison
      rv_current : float — current realized volatility
    """
    from scipy.optimize import brentq

    # Classical IV for comparison
    def bs_price(S, K, r, T, sigma, opt='call'):
        if T <= 0 or sigma <= 0:
            return max(S-K, 0) if opt=='call' else max(K-S, 0)
        d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if opt == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def get_classical_iv(C, S, K, r, T):
        try:
            sigma = np.sqrt(2*np.pi/T)*C/S
            for _ in range(200):
                p = bs_price(S, K, r, T, sigma)
                d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
                v  = S*norm.pdf(d1)*np.sqrt(T)
                if abs(p-C) < 1e-8: return sigma
                if v < 1e-10: break
                sigma -= (p-C)/v
                sigma = max(sigma, 1e-6)
            return float(brentq(
                lambda s: bs_price(S,K,r,T,s)-C,
                1e-6, 10.0))
        except:
            return np.nan

    # Get classical IV for each row and average
    cl_ivs = []
    for _, row in df_window.iterrows():
        iv = get_classical_iv(
            row['C_market'], row['Spot'],
            row['Strike'], row['r'], row['T_years'])
        cl_ivs.append(iv)
    classical_iv = np.nanmean(cl_ivs)

    # PINN IV extraction
    r_mean     = float(df_window['r'].mean())
    rv_mean    = float(df_window['RV_22'].mean()) \
                 if 'RV_22' in df_window.columns else 0.15
    sigma_init = rv_mean if not np.isnan(rv_mean) else 0.15

    if verbose:
        print(f"\n  Extracting PINN IV...")
        print(f"  Window: {df_window['Date'].min().date()} "
              f"to {df_window['Date'].max().date()}")
        print(f"  sigma_init (RV): {sigma_init*100:.2f}%")
        print(f"  Classical IV   : {classical_iv*100:.2f}%")

    pinn = InversePINN(sigma_init=sigma_init, r=r_mean, device=DEVICE)
    pinn.train(df_window, n_epochs=5000, verbose=verbose)

    pinn_iv = pinn.get_pinn_iv()

    return {
        'PINN_IV'       : pinn_iv,
        'Classical_IV'  : classical_iv,
        'RV_current'    : rv_mean,
        'Date'          : df_window['Date'].max(),
        'Spot'          : df_window['Spot'].iloc[-1],
        'sigma_trace'   : pinn.sigma_trace,
    }


# ================================================================
# SECTION 4: MODE B — FUTURE RV FORECASTING
# ================================================================

def forecast_rv(current_pinn_iv,
                history_file='nifty_pinn_iv_results.csv',
                classical_file='nifty_atm_iv_classical.csv'):
    """
    Given today's PINN IV, forecasts next month's realized volatility.

    Uses the regression from Step 4:
      RV_{t+1} = alpha + beta * PINN_IV_t

    Fitted on all available historical PINN IV windows.
    Returns point forecast + 95% prediction interval.
    """
    # Load historical PINN IV results
    df_pinn = pd.read_csv(history_file)
    df_pinn['Date']         = pd.to_datetime(df_pinn['Date'])
    df_pinn['Window_Start'] = pd.to_datetime(df_pinn['Window_Start'])
    df_pinn['Window_End']   = pd.to_datetime(df_pinn['Window_End'])

    df_cl = pd.read_csv(classical_file)
    df_cl['Date'] = pd.to_datetime(df_cl['Date'])

    # Build regression dataset
    records = []
    for _, row in df_pinn.iterrows():
        ws   = row['Window_Start']
        we   = row['Window_End']
        mask = (df_cl['Date'] >= ws) & (df_cl['Date'] <= we)
        win  = df_cl[mask]
        if len(win) == 0:
            continue
        records.append({
            'PINN_IV'    : row['PINN_IV'],
            'Classical_IV': win['IV_classical'].mean(),
            'RV_current' : win['RV_22'].mean(),
        })

    df = pd.DataFrame(records)
    df['RV_next'] = df['RV_current'].shift(-1)
    df = df.dropna()

    if len(df) < 4:
        return {
            'error': 'Insufficient historical data for regression. '
                     'Need at least 4 windows. Download more years.'
        }

    # Fit regression
    y = df['RV_next']
    X = sm.add_constant(df[['PINN_IV']])
    model = sm.OLS(y, X).fit()

    # Point forecast
    X_new = pd.DataFrame({'const': [1.0],
                          'PINN_IV': [current_pinn_iv]})
    pred  = model.get_prediction(X_new)
    pred_summary = pred.summary_frame(alpha=0.05)

    point_forecast = pred_summary['mean'].values[0]
    ci_lower       = pred_summary['obs_ci_lower'].values[0]
    ci_upper       = pred_summary['obs_ci_upper'].values[0]

    return {
        'Input_PINN_IV'        : current_pinn_iv,
        'Predicted_RV_next'    : point_forecast,
        'CI_95_lower'          : max(ci_lower, 0),
        'CI_95_upper'          : ci_upper,
        'Model_R2'             : model.rsquared,
        'Model_beta'           : model.params['PINN_IV'],
        'Model_alpha'          : model.params['const'],
        'Model_p_beta'         : model.pvalues['PINN_IV'],
        'N_observations'       : int(model.nobs),
        'model'                : model,
        'history_df'           : df,
    }


# ================================================================
# SECTION 5: PLOTS
# ================================================================

def plot_extraction(result, outfile='predict_extraction.png'):
    """Plot sigma convergence during PINN extraction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"PINN IV Extraction — {result['Date'].date()}",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(result['sigma_trace'], 'b-', lw=1.5)
    ax.axhline(result['PINN_IV'], color='green', ls='--', lw=1.5,
               label=f"PINN IV = {result['PINN_IV']*100:.2f}%")
    ax.axhline(result['Classical_IV'], color='red', ls=':', lw=1.5,
               label=f"Classical IV = {result['Classical_IV']*100:.2f}%")
    ax.axhline(result['RV_current'], color='darkorange', ls='--', lw=1.2,
               label=f"RV = {result['RV_current']*100:.2f}%")
    ax.set_title('Sigma Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Implied Volatility')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    methods = ['PINN IV', 'Classical IV', 'Realized Vol']
    values  = [result['PINN_IV']*100,
               result['Classical_IV']*100,
               result['RV_current']*100]
    colors  = ['steelblue', 'green', 'darkorange']
    bars    = ax.bar(methods, values, color=colors,
                     alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.1,
                f'{val:.2f}%', ha='center',
                fontsize=11, fontweight='bold')
    ax.set_title('IV Comparison')
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


def plot_forecast(forecast_result, outfile='predict_forecast.png'):
    """Plot forecast with historical context and CI."""
    df  = forecast_result['history_df']
    mdl = forecast_result['model']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('RV Forecasting from PINN IV',
                 fontsize=13, fontweight='bold')

    # Regression plot with prediction
    ax = axes[0]
    iv_range = np.linspace(df['PINN_IV'].min(),
                           df['PINN_IV'].max()*1.1, 100)
    X_range  = sm.add_constant(
        pd.DataFrame({'PINN_IV': iv_range}))
    pred_range = mdl.get_prediction(X_range).summary_frame(alpha=0.05)

    ax.fill_between(iv_range*100,
                    pred_range['obs_ci_lower']*100,
                    pred_range['obs_ci_upper']*100,
                    alpha=0.15, color='steelblue',
                    label='95% Prediction Interval')
    ax.plot(iv_range*100, pred_range['mean']*100,
            'b-', lw=2.0,
            label=f"OLS fit (R²={forecast_result['Model_R2']:.3f})")
    ax.scatter(df['PINN_IV']*100, df['RV_next']*100,
               color='steelblue', s=60, alpha=0.8, zorder=3,
               label='Historical observations')

    # Mark the new prediction
    new_iv = forecast_result['Input_PINN_IV']
    new_rv = forecast_result['Predicted_RV_next']
    ax.scatter([new_iv*100], [new_rv*100],
               color='red', s=150, zorder=5, marker='*',
               label=f"New prediction: {new_rv*100:.2f}%")
    ax.axvline(new_iv*100, color='red', ls='--',
               lw=1.0, alpha=0.5)

    ax.set_xlabel('PINN IV_t (%)')
    ax.set_ylabel('RV_{t+1} (%)')
    ax.set_title('Regression: PINN IV → Future RV')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Forecast summary
    ax = axes[1]
    ax.axis('off')
    summary_text = f"""
FORECAST SUMMARY
{'─'*35}

Input PINN IV     : {new_iv*100:.2f}%

Predicted RV      : {new_rv*100:.2f}%
95% CI Lower      : {forecast_result['CI_95_lower']*100:.2f}%
95% CI Upper      : {forecast_result['CI_95_upper']*100:.2f}%

Model Statistics:
  R²              : {forecast_result['Model_R2']:.4f}
  Beta (IV->RV)   : {forecast_result['Model_beta']:.4f}
  Alpha           : {forecast_result['Model_alpha']*100:.4f}%
  p-value (beta)  : {forecast_result['Model_p_beta']:.4f}
  N observations  : {forecast_result['N_observations']}

Interpretation:
  If PINN IV = {new_iv*100:.1f}%, the model predicts
  next month's RV will be {new_rv*100:.1f}%
  (95% CI: {forecast_result['CI_95_lower']*100:.1f}% to
           {forecast_result['CI_95_upper']*100:.1f}%)
"""
    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue',
                      alpha=0.3))

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PINN IV Prediction Pipeline')
    parser.add_argument('--mode', type=str,
                        default='both',
                        choices=['extract', 'forecast', 'both'],
                        help='extract=Mode A, forecast=Mode B, '
                             'both=A then B')
    args = parser.parse_args()

    print("=" * 65)
    print("IV PREDICTION PIPELINE")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 65)

    extraction_result = None

    # ── MODE A: Extract IV from recent data ──────────────────
    if args.mode in ['extract', 'both']:
        print("\n" + "=" * 65)
        print("MODE A: REAL-TIME IV EXTRACTION")
        print("=" * 65)

        # Load most recent 22 days from your classical IV file
        # In production replace this with live NSE data feed
        data_file = 'nifty_atm_iv_classical.csv'
        if not os.path.exists(data_file):
            data_file = 'nifty_atm_options_baseline.csv'

        df_all = pd.read_csv(data_file)
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date').reset_index(drop=True)
        df_all = df_all.dropna(subset=['C_market', 'Spot',
                                       'Strike', 'T_years',
                                       'Moneyness', 'r'])

        # Take the most recent 22 trading days
        df_window = df_all.tail(22).copy()

        print(f"\n  Using most recent 22 trading days:")
        print(f"  {df_window['Date'].min().date()} to "
              f"{df_window['Date'].max().date()}")
        print(f"  Spot range: {df_window['Spot'].min():.0f} "
              f"to {df_window['Spot'].max():.0f}")

        # Add RV_22 if not present
        if 'RV_22' not in df_window.columns:
            df_window['RV_22'] = 0.12

        extraction_result = extract_iv_today(df_window, verbose=True)

        print(f"\n{'='*50}")
        print(f"MODE A RESULTS")
        print(f"{'='*50}")
        print(f"  Date           : {extraction_result['Date'].date()}")
        print(f"  Nifty Spot     : {extraction_result['Spot']:.0f}")
        print(f"  PINN IV        : "
              f"{extraction_result['PINN_IV']*100:.2f}%")
        print(f"  Classical IV   : "
              f"{extraction_result['Classical_IV']*100:.2f}%")
        print(f"  Realized Vol   : "
              f"{extraction_result['RV_current']*100:.2f}%")
        print(f"  Difference     : "
              f"{abs(extraction_result['PINN_IV']-extraction_result['Classical_IV'])*100:.2f}pp")

        plot_extraction(extraction_result, 'predict_extraction.png')

    # ── MODE B: Forecast future RV ────────────────────────────
    if args.mode in ['forecast', 'both']:
        print("\n" + "=" * 65)
        print("MODE B: FUTURE RV FORECASTING")
        print("=" * 65)

        # Use extracted PINN IV from Mode A
        # or load the most recent from historical results
        if extraction_result is not None:
            current_iv = extraction_result['PINN_IV']
            print(f"\n  Using freshly extracted PINN IV: "
                  f"{current_iv*100:.2f}%")
        else:
            # Load most recent PINN IV from historical results
            df_hist = pd.read_csv('nifty_pinn_iv_results.csv')
            current_iv = df_hist['PINN_IV'].iloc[-1]
            print(f"\n  Using most recent historical PINN IV: "
                  f"{current_iv*100:.2f}%")

        if not os.path.exists('nifty_pinn_iv_results.csv'):
            print("  ERROR: nifty_pinn_iv_results.csv not found.")
            print("  Run step2_pinn_v4.py first.")
        else:
            forecast = forecast_rv(
                current_pinn_iv=current_iv,
                history_file='nifty_pinn_iv_results.csv',
                classical_file='nifty_atm_iv_classical.csv')

            if 'error' in forecast:
                print(f"  {forecast['error']}")
            else:
                print(f"\n{'='*50}")
                print(f"MODE B RESULTS")
                print(f"{'='*50}")
                print(f"  Input PINN IV       : "
                      f"{forecast['Input_PINN_IV']*100:.2f}%")
                print(f"  Predicted RV (next) : "
                      f"{forecast['Predicted_RV_next']*100:.2f}%")
                print(f"  95% CI              : "
                      f"[{forecast['CI_95_lower']*100:.2f}%, "
                      f"{forecast['CI_95_upper']*100:.2f}%]")
                print(f"  Model R²            : "
                      f"{forecast['Model_R2']:.4f}")
                print(f"  Beta (significant)  : "
                      f"{'Yes' if forecast['Model_p_beta']<0.05 else 'No'} "
                      f"(p={forecast['Model_p_beta']:.4f})")

                plot_forecast(forecast, 'predict_forecast.png')

    print("\n" + "=" * 65)
    print("PREDICTION COMPLETE")
    print("=" * 65)
    print("""
Output files:
  predict_extraction.png   <- sigma convergence + IV comparison
  predict_forecast.png     <- regression fit + forecast summary

To predict IV for any specific window:
  Modify df_window in Mode A to use your desired date range.
  The PINN will extract IV for that specific window.

To forecast for a specific PINN IV value:
  Call forecast_rv(current_pinn_iv=0.12) directly.
  Change 0.12 to whatever IV you want to forecast from.
""")