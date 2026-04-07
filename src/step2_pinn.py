"""
================================================================
STEP 2 — PINN v4
================================================================
Fix for sigma converging to wrong value (~2% instead of ~13%):

Root cause: short-dated ATM options (tau=0.03-0.06) give very
weak PDE constraints on sigma because d2c/ds2 is near-zero for
short maturities. Data loss alone finds a degenerate low-sigma
solution that technically fits the prices but is physically wrong.

Fixes applied:
  1. lambda_data raised from 5 to 50 — market price dominates
  2. Sigma prior loss added — weak regularization toward RV
     prevents sigma drifting to degenerate low values
  3. Epochs raised to 5000 for better convergence
  4. Added direct IV loss — penalizes |BS(sigma) - C_market|
     computed analytically, giving sigma a direct gradient signal

INPUT:  nifty_atm_iv_classical.csv
OUTPUT: nifty_pinn_iv_results.csv
        step2_pinn_window_01.png
        step2_pinn_summary.png
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
from scipy.stats import norm as scipy_norm

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ================================================================
# SECTION 1: BLACK-SCHOLES IN PYTORCH (differentiable)
# ================================================================

def bs_call_torch(S, K, r, T, sigma):
    """
    Differentiable BS call price in PyTorch.
    Used to compute direct IV loss with exact gradient to sigma.
    All inputs are torch tensors.
    """
    eps  = 1e-8
    d1   = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + eps)
    d2   = d1 - sigma * torch.sqrt(T)
    # Normal CDF via erf
    n_d1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2)))
    n_d2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2)))
    return S * n_d1 - K * torch.exp(-r * T) * n_d2


# ================================================================
# SECTION 2: NETWORK
# ================================================================

class BSPINN(nn.Module):
    """
    5 layers x 64 neurons, tanh activation.
    softplus output: smooth, always positive, non-saturating.
    Final bias=-4 so initial output ≈ 0.018 ≈ typical c=C/K.
    """
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


# ================================================================
# SECTION 3: PDE AND BOUNDARY CONDITIONS
# ================================================================

def bs_pde_residual(net, tau, s, sigma, r):
    """
    BS PDE residual — tau and s must have requires_grad=True.
    -dc/dtau + 0.5*sigma^2*s^2*(d2c/ds2) + r*s*(dc/ds) - r*c = 0
    """
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
            + 0.5 * sigma**2 * s**2 * d2c_ds2
            + r * s * dc_ds
            - r * c)


def boundary_loss(net, tau_min, tau_max, r,
                  s_max=1.2, n_bc=300, device=DEVICE):
    loss = torch.tensor(0.0, device=device)
    # BC1: terminal payoff
    s1   = torch.rand(n_bc, 1, device=device) * (s_max - 0.5) + 0.5
    tau1 = torch.zeros(n_bc, 1, device=device)
    loss = loss + torch.mean(
        (net(tau1, s1) - torch.relu(s1 - 1.0))**2)
    # BC2: far OTM lower
    tau2 = (torch.rand(n_bc, 1, device=device) *
            (tau_max - tau_min) + tau_min)
    s2   = torch.full((n_bc, 1), 0.5, device=device)
    loss = loss + torch.mean(net(tau2, s2)**2)
    # BC3: deep ITM upper
    tau3 = (torch.rand(n_bc, 1, device=device) *
            (tau_max - tau_min) + tau_min)
    s3   = torch.full((n_bc, 1), s_max, device=device)
    c3   = s3 - torch.exp(-r * tau3)
    loss = loss + torch.mean((net(tau3, s3) - c3)**2)
    return loss


# ================================================================
# SECTION 4: INVERSE PINN
# ================================================================

class InversePINN:
    """
    Inverse PINN with direct IV loss for short-dated options.

    For short-dated ATM options (tau < 0.1), the PDE curvature
    term is small, giving sigma weak gradient signal from PDE alone.

    Solution: add a direct IV loss using the differentiable BS
    formula. This gives sigma a strong, direct gradient:
      L_iv = mean( (BS_price(sigma) - C_market)^2 )

    This is NOT the same as Newton-Raphson:
      - NR finds sigma analytically per option, ignoring PDE
      - Here sigma is shared across all options in the window,
        constrained by both PDE and data simultaneously
      - The PDE still regularizes sigma across the domain

    sigma_prior_loss gives a weak pull toward RV as starting anchor.
    Weight 0.1 means it's a suggestion not a constraint.
    """
    def __init__(self, sigma_init, r, device=DEVICE):
        self.r      = torch.tensor(float(r), device=device)
        self.device = device
        self.net    = BSPINN().to(device)

        self.log_sigma = nn.Parameter(
            torch.tensor(float(np.log(max(sigma_init, 0.01))),
                         dtype=torch.float32, device=device))
        self.log_sigma_init = float(np.log(max(sigma_init, 0.01)))

        self.losses      = {'total': [], 'pde': [], 'bc': [],
                            'data': [], 'iv': []}
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
              lambda_prior=0.1,
              verbose=True):
        """
        Loss = lambda_pde   * L_pde        (physics)
             + lambda_bc    * L_bc         (boundary conditions)
             + lambda_data  * L_data       (network prices vs market)
             + lambda_iv    * L_iv         (BS(sigma) vs market directly)
             + lambda_prior * L_prior      (sigma near RV as weak anchor)

        lambda_iv=100: direct BS pricing error gives sigma
        the strongest possible gradient signal.
        lambda_prior=0.1: weak pull toward RV, prevents
        degenerate solutions without dominating.
        """
        optimizer = optim.Adam([
            {'params': self.net.parameters(), 'lr': lr_net},
            {'params': [self.log_sigma],       'lr': lr_sigma},
        ])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=2000, gamma=0.5)

        # Data tensors
        tau_np = df_window['T_years'].values.astype(np.float32)
        s_np   = df_window['Moneyness'].values.astype(np.float32)
        c_np   = (df_window['C_market'] /
                  df_window['Strike']).values.astype(np.float32)
        # Raw prices and spot/strike for direct BS loss
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

        tau_min = float(tau_np.min()) * 0.5
        tau_max = float(tau_np.max()) * 1.5
        s_lo, s_hi = 0.85, 1.15

        log_rv = self.log_sigma_init  # use initial (=RV) as prior mean

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # PDE loss
            tau_f = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True) *
                     (tau_max - tau_min) + tau_min)
            s_f   = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True) *
                     (s_hi - s_lo) + s_lo)
            loss_pde  = torch.mean(
                bs_pde_residual(self.net, tau_f, s_f,
                                self.sigma, self.r)**2)

            # BC loss
            loss_bc   = boundary_loss(
                self.net, tau_min, tau_max, self.r,
                device=self.device)

            # Network data loss (normalized prices)
            loss_data = torch.mean(
                (self.net(tau_data, s_data) - c_data)**2)

            # Direct IV loss: BS(sigma) vs market price
            # This gives sigma a direct gradient — strongest signal
            C_pred_bs = bs_call_torch(
                S_data, K_data, self.r, T_data, self.sigma)
            loss_iv   = torch.mean((C_pred_bs - C_data)**2)

            # Sigma prior: weak pull toward initial RV estimate
            loss_prior = (self.log_sigma - log_rv)**2

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

            sig_val = self.sigma.item()
            self.losses['total'].append(loss.item())
            self.losses['pde'].append(loss_pde.item())
            self.losses['bc'].append(loss_bc.item())
            self.losses['data'].append(loss_data.item())
            self.losses['iv'].append(loss_iv.item())
            self.sigma_trace.append(sig_val)

            if epoch == 0 and verbose:
                print(f"    Epoch    1 | sigma={sig_val*100:.2f}% | "
                      f"PDE={loss_pde.item():.3e} | "
                      f"IV_loss={loss_iv.item():.3e}")

            if verbose and (epoch + 1) % 500 == 0:
                print(f"    Epoch {epoch+1:4d} | "
                      f"sigma={sig_val*100:.2f}% | "
                      f"Total={loss.item():.3e} | "
                      f"PDE={loss_pde.item():.3e} | "
                      f"IV={loss_iv.item():.3e} | "
                      f"Data={loss_data.item():.3e}")
        return self


# ================================================================
# SECTION 5: FORWARD PINN (validation only)
# ================================================================

class ForwardPINN:
    def __init__(self, sigma, r, device=DEVICE):
        self.sigma  = torch.tensor(float(sigma), device=device)
        self.r      = torch.tensor(float(r),     device=device)
        self.device = device
        self.net    = BSPINN().to(device)
        self.losses = {'total': [], 'pde': [], 'bc': [], 'data': []}

    def train(self, df_window, n_epochs=2000, lr=1e-3,
              n_colloc=2000, lambda_pde=1.0, lambda_bc=10.0,
              lambda_data=50.0, verbose=True):

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.5)

        tau_np = df_window['T_years'].values.astype(np.float32)
        s_np   = df_window['Moneyness'].values.astype(np.float32)
        c_np   = (df_window['C_market'] /
                  df_window['Strike']).values.astype(np.float32)

        tau_data = torch.tensor(tau_np, device=self.device).unsqueeze(1)
        s_data   = torch.tensor(s_np,   device=self.device).unsqueeze(1)
        c_data   = torch.tensor(c_np,   device=self.device).unsqueeze(1)

        tau_min = float(tau_np.min()) * 0.5
        tau_max = float(tau_np.max()) * 1.5
        s_lo, s_hi = 0.85, 1.15

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            tau_f = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True) *
                     (tau_max - tau_min) + tau_min)
            s_f   = (torch.rand(n_colloc, 1, device=self.device,
                                requires_grad=True) *
                     (s_hi - s_lo) + s_lo)

            loss_pde  = torch.mean(
                bs_pde_residual(self.net, tau_f, s_f,
                                self.sigma, self.r)**2)
            loss_bc   = boundary_loss(
                self.net, tau_min, tau_max, self.r,
                device=self.device)
            loss_data = torch.mean(
                (self.net(tau_data, s_data) - c_data)**2)

            loss = (lambda_pde  * loss_pde
                  + lambda_bc   * loss_bc
                  + lambda_data * loss_data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            self.losses['total'].append(loss.item())
            self.losses['pde'].append(loss_pde.item())
            self.losses['bc'].append(loss_bc.item())
            self.losses['data'].append(loss_data.item())

            if epoch == 0 and verbose:
                print(f"    Epoch    1 | PDE={loss_pde.item():.4e}")

            if verbose and (epoch + 1) % 500 == 0:
                print(f"    Epoch {epoch+1:4d} | "
                      f"Total={loss.item():.3e} | "
                      f"PDE={loss_pde.item():.3e} | "
                      f"BC={loss_bc.item():.3e} | "
                      f"Data={loss_data.item():.3e}")
        return self


# ================================================================
# SECTION 6: ROLLING WINDOW
# ================================================================

class RollingInversePINN:
    def __init__(self, df, window_size=22, step_size=22,
                 device=DEVICE):
        self.df          = df.copy().reset_index(drop=True)
        self.window_size = window_size
        self.step_size   = step_size
        self.device      = device
        self.results     = []

    def run(self, n_epochs=5000, verbose=True):
        n      = len(self.df)
        starts = list(range(0, n - self.window_size + 1,
                            self.step_size))
        total  = len(starts)
        print(f"  {total} windows x {n_epochs} epochs")

        for w_idx, start in enumerate(starts):
            end    = start + self.window_size
            df_win = self.df.iloc[start:end].copy()

            mid   = df_win['Date'].iloc[self.window_size // 2]
            r_m   = float(df_win['r'].mean())
            rv_m  = float(df_win['RV_22'].mean())
            s_ini = rv_m if not np.isnan(rv_m) else 0.15

            print(f"\n  [{w_idx+1}/{total}] "
                  f"{df_win['Date'].iloc[0].date()} – "
                  f"{df_win['Date'].iloc[-1].date()} | "
                  f"init={s_ini*100:.1f}%")

            pinn = InversePINN(s_ini, r_m, self.device)
            pinn.train(df_win, n_epochs=n_epochs,
                       verbose=(w_idx == 0))

            iv = pinn.get_pinn_iv()
            cl = (df_win['IV_classical'].mean()
                  if 'IV_classical' in df_win.columns
                  else float('nan'))
            print(f"  PINN={iv*100:.2f}%  "
                  f"Classical={cl*100:.2f}%  "
                  f"RV={rv_m*100:.2f}%")

            self.results.append({
                'Date'        : mid,
                'Window_Start': df_win['Date'].iloc[0],
                'Window_End'  : df_win['Date'].iloc[-1],
                'PINN_IV'     : iv,
                'Classical_IV': cl,
                'RV_22_mean'  : rv_m,
                'r_mean'      : r_m,
                'sigma_init'  : s_ini,
            })

            if w_idx == 0:
                plot_single_window(pinn, df_win,
                                   'step2_pinn_window_01.png')

        return pd.DataFrame(self.results)


# ================================================================
# SECTION 7: PLOTS
# ================================================================

def plot_single_window(pinn, df_win, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Step 2: Inverse PINN — Single Window v4',
                 fontsize=13, fontweight='bold')

    ep = range(1, len(pinn.losses['total']) + 1)
    ax = axes[0, 0]
    ax.semilogy(ep, pinn.losses['total'], 'k-',  lw=1.5, label='Total')
    ax.semilogy(ep, pinn.losses['pde'],   'b-',  lw=1.0, label='PDE')
    ax.semilogy(ep, pinn.losses['bc'],    '-',   lw=1.0,
                color='darkorange', label='BC')
    ax.semilogy(ep, pinn.losses['data'],  'g-',  lw=1.0, label='Data')
    ax.semilogy(ep, pinn.losses['iv'],    'm-',  lw=1.0, label='IV direct')
    ax.set_title('Loss Convergence'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(pinn.sigma_trace, 'b-', lw=1.5)
    rv = df_win['RV_22'].mean()
    ax.axhline(rv, color='darkorange', ls='--',
               label=f'RV={rv*100:.1f}%')
    ax.axhline(pinn.get_pinn_iv(), color='green', ls='--',
               label=f'PINN={pinn.get_pinn_iv()*100:.1f}%')
    if 'IV_classical' in df_win.columns:
        cl = df_win['IV_classical'].mean()
        ax.axhline(cl, color='red', ls=':',
                   label=f'Classical={cl*100:.1f}%')
    ax.set_title('Sigma Convergence'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    pinn.net.eval()
    with torch.no_grad():
        t = torch.tensor(df_win['T_years'].values,
                         dtype=torch.float32,
                         device=pinn.device).unsqueeze(1)
        s = torch.tensor(df_win['Moneyness'].values,
                         dtype=torch.float32,
                         device=pinn.device).unsqueeze(1)
        cp = pinn.net(t, s).cpu().numpy().flatten()
    ca = (df_win['C_market'] / df_win['Strike']).values
    ax.scatter(ca, cp, alpha=0.7, s=30, color='steelblue')
    lim = [min(ca.min(), cp.min()), max(ca.max(), cp.max())]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect fit')
    ax.set_xlabel('Market c=C/K'); ax.set_ylabel('PINN c=C/K')
    ax.set_title('PINN vs Market Prices')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(df_win['Moneyness'], bins=15,
            color='teal', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', ls='--', label='ATM')
    ax.set_xlabel('Moneyness S/K')
    ax.set_title(f'Window Data\nPINN={pinn.get_pinn_iv()*100:.2f}%  '
                 f'Classical={df_win["IV_classical"].mean()*100:.2f}% '
                 if "IV_classical" in df_win.columns else
                 f'Window Data\nPINN={pinn.get_pinn_iv()*100:.2f}%')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Plot: {fname}")
    plt.close()


def plot_summary(df_res, df_cl=None,
                 fname='step2_pinn_summary.png'):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('PINN IV vs Classical IV vs Realized Vol',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(df_res['Date'], df_res['PINN_IV']*100,
            'b-o', lw=1.5, ms=5, label='PINN IV')
    ax.plot(df_res['Date'], df_res['RV_22_mean']*100,
            color='darkorange', lw=1.0, alpha=0.8,
            label='Realized Vol')
    if 'Classical_IV' in df_res.columns:
        ax.plot(df_res['Date'], df_res['Classical_IV']*100,
                'g--', lw=1.0, alpha=0.8, label='Classical IV')
    ax.set_ylabel('Volatility (%)'); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_title('IV Over Time')

    ax = axes[1]
    ax.scatter(df_res['RV_22_mean']*100, df_res['PINN_IV']*100,
               color='steelblue', s=60, alpha=0.8)
    ax.set_xlabel('Realized Vol (%)'); ax.set_ylabel('PINN IV (%)')
    ax.set_title('PINN IV vs Realized Vol')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Plot: {fname}")
    plt.close()


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 2: PINN v4")
    print("=" * 60)

    data_file = 'nifty_atm_iv_classical.csv'
    if not os.path.exists(data_file):
        data_file = 'nifty_atm_options_baseline.csv'

    print(f"\n[1/3] Loading: {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['C_market','Spot','Strike',
                           'T_years','Moneyness','r','RV_22'])
    print(f"  {len(df)} rows | "
          f"{df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")

    WINDOW = 22

    # STAGE 1
    print("\n[2/3] STAGE 1: single window validation...")
    df_d   = df.iloc[:WINDOW].copy()
    r_m    = float(df_d['r'].mean())
    rv_m   = float(df_d['RV_22'].mean())
    s_init = rv_m if not np.isnan(rv_m) else 0.15
    cl_iv  = (df_d['IV_classical'].mean()
              if 'IV_classical' in df_d.columns else float('nan'))

    print(f"  sigma_init={s_init*100:.2f}%  "
          f"r={r_m*100:.2f}%  "
          f"classical_iv={cl_iv*100:.2f}%")

    print("\n  -- Forward PINN --")
    fwd = ForwardPINN(sigma=s_init, r=r_m, device=DEVICE)
    fwd.train(df_d, n_epochs=2000, verbose=True)

    print("\n  -- Inverse PINN --")
    inv = InversePINN(sigma_init=s_init, r=r_m, device=DEVICE)
    inv.train(df_d, n_epochs=5000, verbose=True)

    pinn_iv = inv.get_pinn_iv()
    print(f"\n  STAGE 1 RESULTS:")
    print(f"  PINN IV     : {pinn_iv*100:.2f}%")
    print(f"  Classical IV: {cl_iv*100:.2f}%")
    print(f"  Realized Vol: {rv_m*100:.2f}%")
    print(f"  Difference  : {abs(pinn_iv - cl_iv)*100:.2f}pp")

    plot_single_window(inv, df_d, 'step2_pinn_window_01.png')

    # STAGE 2 — uncomment when Stage 1 converges near classical IV
    print("\n[3/3] STAGE 2: rolling window (uncomment to run)...")

    rolling = RollingInversePINN(df, window_size=WINDOW,
                                 step_size=WINDOW, device=DEVICE)
    df_res = rolling.run(n_epochs=5000)
    df_res.to_csv('nifty_pinn_iv_results.csv', index=False)
    print(f"  Saved: nifty_pinn_iv_results.csv")
    print(f"  Mean PINN IV  : {df_res['PINN_IV'].mean()*100:.2f}%")
    print(f"  Mean Classical: {df_res['Classical_IV'].mean()*100:.2f}%")
    plot_summary(df_res, fname='step2_pinn_summary.png')

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print("""
PINN IV should now be close to Classical IV (~13%).
If the difference is < 2 percentage points, Stage 1 is validated.
Uncomment Stage 2 to run the full rolling window.
""")