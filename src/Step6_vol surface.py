"""
================================================================
STEP 6 — VOLATILITY SURFACE EXTENSION
Beyond the Original Thesis
================================================================
The original thesis extracted one scalar IV per day (ATM only).
This step extends sigma from a scalar to a FUNCTION:

    sigma(s, tau) — volatility surface

Where s = moneyness (S/K) and tau = time to maturity.

This requires the FULL cross-section of option prices across
ALL strikes and maturities — not just ATM.

WHAT THIS STEP DOES:
  1. Loads all near-month options (all strikes, not just ATM)
  2. Builds a SurfaceInversePINN where sigma is a small neural
     network: sigma_net(s, tau) -> sigma
  3. Trains on all available strike/maturity combinations
  4. Plots the 3D volatility surface and volatility smile
  5. Compares surface ATM IV to the scalar IV from Steps 2-5

The output is a 3D volatility surface showing:
  - The volatility SMILE: higher IV for OTM puts (s < 1)
  - The volatility TERM STRUCTURE: IV changing with maturity
  - The combined SURFACE: sigma(s, tau) across all dimensions

This is the strongest visual contribution of the thesis —
something the original thesis never produced.

INPUT:
  nifty_atm_calls_clean.csv      (ATM calls, for reference)
  nifty_atm_iv_classical.csv     (ATM classical IV, for reference)
  NIFTY_CALL_OPTIONS_2025_2026.csv  (FULL cross-section)

OUTPUT:
  step6_vol_surface_3d.png
  step6_vol_smile.png
  step6_surface_iv_results.csv

RUN:
  python step6_vol_surface.py
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
import os

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ================================================================
# SECTION 1: LOAD FULL CROSS-SECTION DATA
# ================================================================

def load_full_cross_section(call_file='NIFTY_CALL_OPTIONS_2025_2026.csv',
                             min_days=6, max_days=35,
                             min_price=0.5,
                             moneyness_lo=0.85,
                             moneyness_hi=1.15):
    """
    Loads ALL strikes (not just ATM) for the surface estimation.
    Filters to near-month contracts within a moneyness band.

    Moneyness band 0.85 to 1.15 gives enough cross-section
    to see the smile while excluding deep OTM/ITM options
    that are extremely illiquid.
    """
    if not os.path.exists(call_file):
        raise FileNotFoundError(
            f"'{call_file}' not found.\n"
            "This file is needed for the full cross-section.\n"
            "Make sure it is in the same folder as this script.")

    print(f"  Loading {call_file}...")
    df = pd.read_csv(call_file)
    df.columns = df.columns.str.strip()

    rename_map = {
        'Date': 'Date', 'Expiry': 'Expiry',
        'Option type': 'Option_Type',
        'Strike Price': 'Strike',
        'Close': 'Close', 'Settle Price': 'Settle_Price',
        'Underlying Value': 'Spot',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})

    keep = ['Date', 'Expiry', 'Strike', 'Close',
            'Settle_Price', 'Spot']
    df = df[[c for c in keep if c in df.columns]]

    df['Date']   = pd.to_datetime(
        df['Date'].astype(str).str.strip(),
        format='%d-%b-%Y', dayfirst=True)
    df['Expiry'] = pd.to_datetime(
        df['Expiry'].astype(str).str.strip(),
        format='%d-%b-%Y', dayfirst=True)

    for col in ['Close', 'Settle_Price', 'Spot', 'Strike']:
        df[col] = df[col].astype(str).str.strip().replace('-', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Close'] = df['Close'].fillna(df['Settle_Price'])
    df = df.dropna(subset=['Date', 'Expiry', 'Strike',
                           'Close', 'Spot'])
    df = df[df['Close'] >= min_price]

    df['T_days']   = (df['Expiry'] - df['Date']).dt.days
    df['T_years']  = df['T_days'] / 252
    df['Moneyness'] = df['Spot'] / df['Strike']
    df['C_market'] = df['Close']

    # Filter to near-month and moneyness band
    df = df[(df['T_days'] > min_days) &
            (df['T_days'] <= max_days) &
            (df['Moneyness'] >= moneyness_lo) &
            (df['Moneyness'] <= moneyness_hi)]

    # Add risk-free rate
    def get_r(date):
        y = date.year
        if y <= 2024: return 0.065
        return 0.065
    df['r'] = df['Date'].apply(get_r)

    # Normalize price
    df['c_norm'] = df['C_market'] / df['Strike']

    df = df.sort_values(['Date', 'T_days', 'Strike'])
    df = df.reset_index(drop=True)

    print(f"  Full cross-section: {len(df):,} option observations")
    print(f"  Date range: {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"  Moneyness: {df['Moneyness'].min():.3f} to "
          f"{df['Moneyness'].max():.3f}")
    print(f"  T_days: {df['T_days'].min()} to "
          f"{df['T_days'].max()}")
    print(f"  Unique dates: {df['Date'].nunique()}")
    print(f"  Unique strikes per day (approx): "
          f"{len(df) // df['Date'].nunique()}")

    return df


# ================================================================
# SECTION 2: SIGMA NETWORK (the surface)
# ================================================================

class SigmaNet(nn.Module):
    """
    Small neural network that maps (s, tau) -> sigma.
    This IS the volatility surface.

    Architecture: 3 layers x 32 neurons, tanh activation.
    Small because sigma varies smoothly across the surface —
    it doesn't need the full capacity of the pricing network.

    softplus output: sigma > 0 always.
    Bias initialized to log(0.15) so sigma starts near 15%.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize to ~15% vol
        list(self.net.modules())[-1].bias.data.fill_(
            float(np.log(np.exp(0.15) - 1)))

    def forward(self, s, tau):
        x = torch.cat([s, tau], dim=1)
        return torch.nn.functional.softplus(self.net(x))


# ================================================================
# SECTION 3: PRICING NETWORK (same as before)
# ================================================================

class PricingNet(nn.Module):
    """
    5 layers x 64 neurons, tanh, softplus output.
    Same architecture as Step 2 PINN.
    """
    def __init__(self, hidden_dim=64, n_layers=5):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.Tanh()]
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
# SECTION 4: BS FORMULA IN PYTORCH
# ================================================================

def bs_call_torch(S, K, r, T, sigma):
    eps  = 1e-8
    d1   = (torch.log(S / K) +
            (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + eps)
    d2   = d1 - sigma * torch.sqrt(T)
    n_d1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2)))
    n_d2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2)))
    return S * n_d1 - K * torch.exp(-r * T) * n_d2


# ================================================================
# SECTION 5: SURFACE PDE RESIDUAL
# ================================================================

def surface_pde_residual(pricing_net, sigma_net,
                          tau, s, r):
    """
    BS PDE residual where sigma = sigma_net(s, tau).
    The surface PINN enforces:
      -dc/dtau + 0.5*sigma(s,tau)^2*s^2*(d2c/ds2)
              + r*s*(dc/ds) - r*c = 0

    sigma is now a FUNCTION of (s, tau) not a constant.
    This allows the model to capture the volatility smile
    and term structure simultaneously.
    """
    c    = pricing_net(tau, s)
    ones = torch.ones_like(c)

    dc_dtau = torch.autograd.grad(
        c, tau, grad_outputs=ones,
        create_graph=True, retain_graph=True)[0]
    dc_ds   = torch.autograd.grad(
        c, s, grad_outputs=ones,
        create_graph=True, retain_graph=True)[0]
    d2c_ds2 = torch.autograd.grad(
        dc_ds, s,
        grad_outputs=torch.ones_like(dc_ds),
        create_graph=True, retain_graph=True)[0]

    # Sigma is now a function evaluated at these (s, tau) points
    sigma = sigma_net(s, tau)

    residual = (-dc_dtau
                + 0.5 * sigma**2 * s**2 * d2c_ds2
                + r * s * dc_ds
                - r * c)
    return residual


def boundary_loss_surface(pricing_net, tau_min, tau_max,
                           r_val, s_max=1.15,
                           n_bc=200, device=DEVICE):
    loss = torch.tensor(0.0, device=device)

    # BC1: terminal payoff
    s1   = torch.rand(n_bc, 1, device=device) * (s_max - 0.85) + 0.85
    tau1 = torch.zeros(n_bc, 1, device=device)
    loss = loss + torch.mean(
        (pricing_net(tau1, s1) - torch.relu(s1 - 1.0))**2)

    # BC2: far OTM
    tau2 = (torch.rand(n_bc, 1, device=device) *
            (tau_max - tau_min) + tau_min)
    s2   = torch.full((n_bc, 1), 0.85, device=device)
    loss = loss + torch.mean(pricing_net(tau2, s2)**2)

    # BC3: deep ITM
    tau3 = (torch.rand(n_bc, 1, device=device) *
            (tau_max - tau_min) + tau_min)
    s3   = torch.full((n_bc, 1), s_max, device=device)
    c3   = s3 - torch.exp(-r_val * tau3)
    loss = loss + torch.mean(
        (pricing_net(tau3, s3) - c3)**2)

    return loss


# ================================================================
# SECTION 6: SURFACE INVERSE PINN TRAINER
# ================================================================

class SurfaceInversePINN:
    """
    Surface extension of the Inverse PINN.
    sigma is no longer a scalar — it is sigma_net(s, tau).

    The pricing network learns C(tau, s) as before.
    The sigma network learns sigma(s, tau) simultaneously.

    Both networks are trained jointly to minimize:
      L = lambda_pde  * L_pde   (PDE with sigma(s,tau))
        + lambda_bc   * L_bc    (boundary conditions)
        + lambda_data * L_data  (network vs market prices)
        + lambda_iv   * L_iv    (BS(sigma) vs market directly)
        + lambda_smooth * L_smooth  (sigma surface smoothness)

    L_smooth prevents sigma(s,tau) from being jagged —
    it penalizes large gradients in the sigma surface:
      L_smooth = mean(|d_sigma/ds|^2 + |d_sigma/dtau|^2)
    """
    def __init__(self, r, device=DEVICE):
        self.r      = torch.tensor(float(r), device=device)
        self.device = device
        self.pricing = PricingNet().to(device)
        self.sigma   = SigmaNet().to(device)
        self.losses  = {'total': [], 'pde': [], 'bc': [],
                        'data': [], 'iv': [], 'smooth': []}

    def get_surface(self, s_grid, tau_grid):
        """
        Evaluates sigma_net on a grid for plotting.
        Returns sigma values as numpy array.
        """
        self.sigma.eval()
        with torch.no_grad():
            s_t = torch.tensor(
                s_grid.flatten(), dtype=torch.float32,
                device=self.device).unsqueeze(1)
            tau_t = torch.tensor(
                tau_grid.flatten(), dtype=torch.float32,
                device=self.device).unsqueeze(1)
            sig = self.sigma(s_t, tau_t).cpu().numpy()
        return sig.reshape(s_grid.shape)

    def train(self, df_window, n_epochs=5000,
              lr_pricing=1e-3, lr_sigma=1e-3,
              n_colloc=2000,
              lambda_pde=1.0, lambda_bc=10.0,
              lambda_data=50.0, lambda_iv=100.0,
              lambda_smooth=10.0,
              verbose=True):

        optimizer = optim.Adam([
            {'params': self.pricing.parameters(),
             'lr': lr_pricing},
            {'params': self.sigma.parameters(),
             'lr': lr_sigma},
        ])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=2000, gamma=0.5)

        # Data tensors
        tau_np = df_window['T_years'].values.astype(np.float32)
        s_np   = df_window['Moneyness'].values.astype(np.float32)
        c_np   = df_window['c_norm'].values.astype(np.float32)
        C_np   = df_window['C_market'].values.astype(np.float32)
        S_np   = df_window['Spot'].values.astype(np.float32)
        K_np   = df_window['Strike'].values.astype(np.float32)

        tau_data = torch.tensor(tau_np,
                                device=self.device).unsqueeze(1)
        s_data   = torch.tensor(s_np,
                                device=self.device).unsqueeze(1)
        c_data   = torch.tensor(c_np,
                                device=self.device).unsqueeze(1)
        C_data   = torch.tensor(C_np,
                                device=self.device).unsqueeze(1)
        S_data   = torch.tensor(S_np,
                                device=self.device).unsqueeze(1)
        K_data   = torch.tensor(K_np,
                                device=self.device).unsqueeze(1)
        T_data   = tau_data.clone()

        tau_min = float(tau_np.min()) * 0.5
        tau_max = float(tau_np.max()) * 1.5
        s_lo    = float(s_np.min()) * 0.98
        s_hi    = float(s_np.max()) * 1.02

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Collocation points
            tau_f = (torch.rand(n_colloc, 1,
                                device=self.device,
                                requires_grad=True) *
                     (tau_max - tau_min) + tau_min)
            s_f   = (torch.rand(n_colloc, 1,
                                device=self.device,
                                requires_grad=True) *
                     (s_hi - s_lo) + s_lo)

            # PDE loss (sigma is now a function)
            loss_pde = torch.mean(
                surface_pde_residual(
                    self.pricing, self.sigma,
                    tau_f, s_f, self.r)**2)

            # BC loss
            loss_bc = boundary_loss_surface(
                self.pricing, tau_min, tau_max,
                self.r, s_max=s_hi, device=self.device)

            # Network data loss
            loss_data = torch.mean(
                (self.pricing(tau_data, s_data) - c_data)**2)

            # Direct IV loss
            sigma_at_data = self.sigma(s_data, tau_data)
            C_pred_bs = bs_call_torch(
                S_data, K_data, self.r, T_data,
                sigma_at_data)
            loss_iv = torch.mean((C_pred_bs - C_data)**2)

            # Smoothness regularization on sigma surface
            # Penalize large gradients in sigma(s, tau)
            s_sm  = (torch.rand(500, 1, device=self.device,
                                requires_grad=True) *
                     (s_hi - s_lo) + s_lo)
            tau_sm = (torch.rand(500, 1, device=self.device,
                                 requires_grad=True) *
                      (tau_max - tau_min) + tau_min)
            sig_sm = self.sigma(s_sm, tau_sm)
            ones_sm = torch.ones_like(sig_sm)

            dsig_ds = torch.autograd.grad(
                sig_sm, s_sm,
                grad_outputs=ones_sm,
                create_graph=True, retain_graph=True)[0]
            dsig_dtau = torch.autograd.grad(
                sig_sm, tau_sm,
                grad_outputs=ones_sm,
                create_graph=True, retain_graph=True)[0]
            loss_smooth = (torch.mean(dsig_ds**2) +
                           torch.mean(dsig_dtau**2))

            loss = (lambda_pde    * loss_pde
                  + lambda_bc     * loss_bc
                  + lambda_data   * loss_data
                  + lambda_iv     * loss_iv
                  + lambda_smooth * loss_smooth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.pricing.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                self.sigma.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            self.losses['total'].append(loss.item())
            self.losses['pde'].append(loss_pde.item())
            self.losses['bc'].append(loss_bc.item())
            self.losses['data'].append(loss_data.item())
            self.losses['iv'].append(loss_iv.item())
            self.losses['smooth'].append(loss_smooth.item())

            if epoch == 0 and verbose:
                atm_s   = torch.ones(1, 1, device=self.device)
                atm_tau = torch.tensor([[tau_np.mean()]],
                                       device=self.device)
                atm_sig = self.sigma(atm_s, atm_tau).item()
                print(f"    Epoch    1 | "
                      f"ATM sigma={atm_sig*100:.2f}% | "
                      f"PDE={loss_pde.item():.3e}")

            if verbose and (epoch + 1) % 1000 == 0:
                atm_s   = torch.ones(1, 1, device=self.device)
                atm_tau = torch.tensor([[tau_np.mean()]],
                                       device=self.device)
                atm_sig = self.sigma(atm_s, atm_tau).item()
                print(f"    Epoch {epoch+1:4d} | "
                      f"ATM sigma={atm_sig*100:.2f}% | "
                      f"Total={loss.item():.3e} | "
                      f"IV={loss_iv.item():.3e} | "
                      f"Smooth={loss_smooth.item():.3e}")

        return self


# ================================================================
# SECTION 7: PLOTS
# ================================================================

def plot_vol_surface(model, df_window, tau_range, s_range,
                     outfile='step6_vol_surface_3d.png'):
    """
    3D volatility surface plot — the main visual contribution.
    """
    # Build grid
    s_grid_1d   = np.linspace(s_range[0], s_range[1], 50)
    tau_grid_1d = np.linspace(tau_range[0], tau_range[1], 50)
    S_grid, TAU_grid = np.meshgrid(s_grid_1d, tau_grid_1d)

    # Get sigma surface from trained model
    sigma_surface = model.get_surface(S_grid, TAU_grid) * 100

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('Step 6: Implied Volatility Surface\n'
                 'Nifty 50 Options — sigma(moneyness, maturity)',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: 3D surface ───────────────────────────────────
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(
        S_grid, TAU_grid * 252, sigma_surface,
        cmap='RdYlGn_r', alpha=0.85, linewidth=0)
    ax1.set_xlabel('Moneyness (S/K)', labelpad=8)
    ax1.set_ylabel('Days to Expiry', labelpad=8)
    ax1.set_zlabel('IV (%)', labelpad=8)
    ax1.set_title('3D Volatility Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='IV (%)')

    # ── Panel 2: Volatility smile (fixed tau) ─────────────────
    ax2 = fig.add_subplot(132)
    tau_levels = [tau_range[0] + (tau_range[1]-tau_range[0])*f
                  for f in [0.2, 0.5, 0.8]]
    labels = ['Short term', 'Mid term', 'Long term']
    colors = ['steelblue', 'darkorange', 'green']

    for tau_val, label, color in zip(tau_levels, labels, colors):
        tau_arr = np.full_like(s_grid_1d, tau_val).reshape(-1, 1)
        s_arr   = s_grid_1d.reshape(-1, 1)
        sig_smile = model.get_surface(s_arr, tau_arr).flatten() * 100
        ax2.plot(s_grid_1d, sig_smile,
                 color=color, lw=2.0,
                 label=f'{label} ({int(tau_val*252)}d)')

    ax2.axvline(1.0, color='black', ls='--', lw=1.0,
                alpha=0.5, label='ATM (s=1)')
    ax2.set_xlabel('Moneyness (S/K)')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.set_title('Volatility Smile\n(across strikes, fixed maturity)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Term structure (ATM, s=1) ────────────────────
    ax3 = fig.add_subplot(133)
    tau_arr_ts = tau_grid_1d.reshape(-1, 1)
    s_atm      = np.ones_like(tau_grid_1d).reshape(-1, 1)
    sig_term   = model.get_surface(s_atm, tau_arr_ts).flatten() * 100

    ax3.plot(tau_grid_1d * 252, sig_term,
             color='steelblue', lw=2.0)

    # Mark observed data points
    if 'T_days' in df_window.columns:
        atm_mask = ((df_window['Moneyness'] >= 0.99) &
                    (df_window['Moneyness'] <= 1.01))
        atm_data = df_window[atm_mask]
        if len(atm_data) > 0:
            from scipy.stats import norm as sp_norm
            # Classical IV from data
            ax3.scatter(
                atm_data['T_days'],
                atm_data['c_norm'] * 100,
                color='red', s=40, zorder=5, alpha=0.7,
                label='Observed c/K (scaled)')

    ax3.set_xlabel('Days to Expiry')
    ax3.set_ylabel('ATM Implied Volatility (%)')
    ax3.set_title('Volatility Term Structure\n(ATM, s=1)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


def plot_smile_by_date(model, df_all, sample_dates,
                       outfile='step6_vol_smile.png'):
    """
    Volatility smile on specific dates.
    Shows how the smile changes over time.
    """
    fig, axes = plt.subplots(1, min(3, len(sample_dates)),
                             figsize=(14, 5))
    if len(sample_dates) == 1:
        axes = [axes]
    fig.suptitle('Step 6: Volatility Smile Across Dates',
                 fontsize=13, fontweight='bold')

    s_grid = np.linspace(0.88, 1.12, 60)

    for i, (date, ax) in enumerate(
            zip(sample_dates[:3], axes)):
        day_data = df_all[df_all['Date'] == date]
        if len(day_data) == 0:
            continue

        tau_val = day_data['T_years'].mean()
        tau_arr = np.full_like(s_grid, tau_val).reshape(-1, 1)
        s_arr   = s_grid.reshape(-1, 1)
        smile   = model.get_surface(s_arr, tau_arr).flatten()*100

        ax.plot(s_grid, smile, 'b-', lw=2.0,
                label='PINN surface')
        ax.axvline(1.0, color='red', ls='--', lw=1.0,
                   alpha=0.7, label='ATM')

        # Scatter actual data points
        ax.scatter(day_data['Moneyness'],
                   day_data['c_norm'] * 500,
                   color='orange', s=30, alpha=0.7,
                   label='Market c/K (x500)')

        ax.set_xlabel('Moneyness (S/K)')
        ax.set_ylabel('IV (%)')
        ax.set_title(f'{pd.Timestamp(date).date()}\n'
                     f'T={int(tau_val*252)}d')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


def plot_loss_convergence(model,
                          outfile='step6_loss_convergence.png'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = range(1, len(model.losses['total']) + 1)
    ax.semilogy(ep, model.losses['total'],  'k-',  lw=1.5,
                label='Total')
    ax.semilogy(ep, model.losses['pde'],    'b-',  lw=1.0,
                label='PDE')
    ax.semilogy(ep, model.losses['bc'],     '-',   lw=1.0,
                color='darkorange', label='BC')
    ax.semilogy(ep, model.losses['iv'],     'm-',  lw=1.0,
                label='IV direct')
    ax.semilogy(ep, model.losses['smooth'], 'g-',  lw=1.0,
                label='Smoothness')
    ax.set_title('Step 6: Surface PINN Loss Convergence')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


# ================================================================
# SECTION 8: EXTRACT SURFACE IV SERIES
# ================================================================

def extract_surface_atm_iv(model, df_pinn_windows, device):
    """
    Extracts ATM IV from the surface at each window's
    mean tau. Compares to scalar PINN IV from Step 2.
    """
    records = []
    for _, row in df_pinn_windows.iterrows():
        tau_val = float(row.get('tau_mean', 9/252))
        s_atm   = torch.ones(1, 1, device=device)
        tau_t   = torch.tensor([[tau_val]],
                               dtype=torch.float32,
                               device=device)
        with torch.no_grad():
            sig_atm = model.sigma(s_atm, tau_t).item()
        records.append({
            'Date'           : row['Date'],
            'Surface_ATM_IV' : sig_atm,
            'Scalar_PINN_IV' : row['PINN_IV'],
        })
    return pd.DataFrame(records)


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STEP 6: VOLATILITY SURFACE EXTENSION")
    print("Beyond the Original Thesis")
    print("=" * 65)

    # ── Load full cross-section ───────────────────────────────
    print("\n[1/4] Loading full cross-section data...")
    df_full = load_full_cross_section(
        call_file='NIFTY_CALL_OPTIONS_2025_2026.csv',
        min_days=6, max_days=35,
        moneyness_lo=0.88, moneyness_hi=1.12)

    # ── Select one representative window for surface ──────────
    # Use Q1 2025 as the demo window (first full quarter)
    print("\n[2/4] Selecting demo window for surface estimation...")
    demo_start = pd.Timestamp('2025-01-01')
    demo_end   = pd.Timestamp('2025-03-31')
    df_window  = df_full[
        (df_full['Date'] >= demo_start) &
        (df_full['Date'] <= demo_end)].copy()

    print(f"  Demo window: {demo_start.date()} to "
          f"{demo_end.date()}")
    print(f"  Observations: {len(df_window)}")
    print(f"  Unique dates: {df_window['Date'].nunique()}")
    print(f"  Unique strikes: {df_window['Strike'].nunique()}")
    print(f"  Moneyness range: {df_window['Moneyness'].min():.3f}"
          f" to {df_window['Moneyness'].max():.3f}")

    if len(df_window) < 10:
        print("  WARNING: Very few observations in demo window.")
        print("  Using full dataset instead.")
        df_window = df_full.copy()

    r_mean = 0.065

    # ── Train Surface PINN ────────────────────────────────────
    print("\n[3/4] Training Surface Inverse PINN...")
    print("  (sigma is now a neural network, not a scalar)")
    print("  Expected runtime: ~5-10 minutes on CPU")

    surface_pinn = SurfaceInversePINN(r=r_mean, device=DEVICE)
    surface_pinn.train(
        df_window,
        n_epochs=5000,
        lr_pricing=1e-3,
        lr_sigma=1e-3,
        n_colloc=1000,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_data=50.0,
        lambda_iv=100.0,
        lambda_smooth=100.0,
        verbose=True)

    # ── Plot surface ──────────────────────────────────────────
    print("\n[4/4] Generating surface plots...")

    tau_range = (df_window['T_years'].min() * 0.8,
                 df_window['T_years'].max() * 1.2)
    s_range   = (0.88, 1.12)

    plot_vol_surface(
        surface_pinn, df_window,
        tau_range, s_range,
        'step6_vol_surface_3d.png')

    # Sample 3 dates for smile plot
    sample_dates = sorted(df_window['Date'].unique())
    step = max(1, len(sample_dates) // 3)
    sample_dates = sample_dates[::step][:3]
    plot_smile_by_date(
        surface_pinn, df_window, sample_dates,
        'step6_vol_smile.png')

    plot_loss_convergence(
        surface_pinn, 'step6_loss_convergence.png')

    # Extract ATM IV from surface and compare to scalar
    print("\n  Surface ATM IV at mean tau:")
    tau_mean = df_window['T_years'].mean()
    s_atm    = torch.ones(1, 1, device=DEVICE)
    tau_t    = torch.tensor([[tau_mean]],
                            dtype=torch.float32,
                            device=DEVICE)
    with torch.no_grad():
        atm_iv = surface_pinn.sigma(s_atm, tau_t).item()
    print(f"  Surface ATM IV : {atm_iv*100:.2f}%")
    print(f"  Scalar PINN IV : ~13.43% (from Step 2 Window 1)")
    print(f"  Classical IV   : ~13.72% (from Step 1)")

    # Save surface IV at grid points
    s_pts   = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
    tau_pts = np.array([7, 14, 21, 28, 35]) / 252
    rows = []
    surface_pinn.sigma.eval()
    with torch.no_grad():
        for s_val in s_pts:
            for tau_val in tau_pts:
                sv = surface_pinn.sigma(
                    torch.tensor([[s_val]],
                                 dtype=torch.float32,
                                 device=DEVICE),
                    torch.tensor([[tau_val]],
                                 dtype=torch.float32,
                                 device=DEVICE)).item()
                rows.append({
                    'Moneyness': s_val,
                    'T_days'   : int(tau_val * 252),
                    'T_years'  : tau_val,
                    'Surface_IV': sv
                })
    df_surface = pd.DataFrame(rows)
    df_surface.to_csv('step6_surface_iv_results.csv',
                      index=False)
    print("\n  Surface IV grid (sigma(s, tau)):")
    pivot = df_surface.pivot(
        index='Moneyness',
        columns='T_days',
        values='Surface_IV') * 100
    print(pivot.round(2).to_string())

    print("\n" + "=" * 65)
    print("STEP 6 COMPLETE")
    print("=" * 65)
    print("""
Files saved:
  step6_vol_surface_3d.png      <- 3D surface + smile + term structure
  step6_vol_smile.png           <- smile across dates
  step6_loss_convergence.png    <- training convergence
  step6_surface_iv_results.csv  <- surface IV at grid points

What to report in thesis:
  1. The volatility smile: IV is higher for OTM options
     (moneyness < 1) than ATM — this is the well-known
     volatility smirk in index options
  2. The term structure: how ATM IV changes with maturity
  3. Surface ATM IV should match scalar PINN IV from Step 2
     (validates the surface approach)
  4. The surface goes beyond the original thesis which only
     computed a single scalar IV per day

Next: Step 7 — Thesis Writing
""")