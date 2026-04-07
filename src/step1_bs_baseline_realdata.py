"""
================================================================
STEP 1 — CLASSICAL BLACK-SCHOLES BASELINE
Real Nifty ATM Options Data
================================================================
Replicates thesis methodology (TH8745, Panda 2015):
Newton-Raphson + Brent IV extraction from real NSE data.

INPUT:
  nifty_atm_options_baseline.csv   (from step0_data_cleaning.py)

OUTPUT:
  nifty_atm_iv_classical.csv       (classical IV time series)
  step1_bs_baseline_realdata.png

RUN:
  pip install numpy scipy pandas matplotlib
  python step1_bs_baseline_realdata.py
================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1: BLACK-SCHOLES FUNCTIONS
# ================================================================

def black_scholes(S, K, r, T, sigma, option_type='call'):
    """
    Black-Scholes European option price.
    Thesis equations 3.12 (call) and 3.13 (put).
    """
    if T <= 1e-6 or sigma <= 1e-6:
        if option_type == 'call':
            return max(S - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)
    else:
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.0)


def bs_vega(S, K, r, T, sigma):
    """Vega = dC/d_sigma. Denominator in Newton-Raphson."""
    if T <= 1e-6 or sigma <= 1e-6:
        return 1e-10
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(market_price, S, K, r, T,
                       option_type='call', tol=1e-8, max_iter=200):
    """
    Extract IV via Newton-Raphson with Brent fallback.
    Returns np.nan if extraction fails.
    """
    # No-arbitrage lower bound check
    if option_type == 'call':
        lb = max(S - K * np.exp(-r * T), 0.0)
    else:
        lb = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < lb - 0.01 or market_price <= 0:
        return np.nan

    # Brenner-Subrahmanyam initial guess
    sigma = np.sqrt(2 * np.pi / max(T, 1e-6)) * market_price / S
    sigma = np.clip(sigma, 0.01, 5.0)

    # Newton-Raphson
    for _ in range(max_iter):
        price = black_scholes(S, K, r, T, sigma, option_type)
        vega  = bs_vega(S, K, r, T, sigma)
        diff  = price - market_price
        if abs(diff) < tol:
            return float(sigma)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma  = max(sigma, 1e-6)

    # Brent fallback
    try:
        def f(s):
            return black_scholes(S, K, r, T, s, option_type) - market_price
        return float(brentq(f, 1e-6, 10.0, xtol=tol, maxiter=max_iter))
    except Exception:
        return np.nan

# ================================================================
# SECTION 2: LOAD DATA
# ================================================================

def load_data(filepath='nifty_atm_options_baseline.csv'):
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found. Run step0_data_cleaning.py first.")

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"  Loaded: {len(df)} rows")
    print(f"  Dates : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Cols  : {df.columns.tolist()}")
    return df

# ================================================================
# SECTION 3: EXTRACT CLASSICAL IV
# ================================================================

def extract_iv(df):
    """Newton-Raphson IV extraction on every row."""
    iv_c, iv_p = [], []

    for i, row in df.iterrows():
        S, K, r, T = row['Spot'], row['Strike'], row['r'], row['T_years']

        # Call IV
        iv_c.append(implied_volatility(
            row['C_market'], S, K, r, T, 'call'))

        # Put IV (if available)
        if 'C_market_PE' in df.columns and not pd.isna(row.get('C_market_PE', np.nan)):
            iv_p.append(implied_volatility(
                row['C_market_PE'], S, K, r, T, 'put'))
        else:
            iv_p.append(np.nan)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(df)} processed...")

    df['IV_call'] = iv_c
    df['IV_put']  = iv_p

    # Average CE and PE (thesis approach)
    df['IV_classical'] = np.where(
        ~np.isnan(df['IV_put']),
        (df['IV_call'] + df['IV_put']) / 2,
        df['IV_call'])

    # Round-trip reprice to validate
    df['C_reprice'] = [
        black_scholes(r['Spot'], r['Strike'], r['r'], r['T_years'],
                      r['IV_call'] if not np.isnan(r['IV_call']) else 0.2,
                      'call')
        for _, r in df.iterrows()]
    df['Reprice_Error'] = abs(df['C_reprice'] - df['C_market'])

    ok = df['IV_classical'].notna().sum()
    print(f"\n  Success: {ok}/{len(df)} = {100*ok/len(df):.1f}%")
    print(f"  Mean IV: {df['IV_classical'].mean()*100:.2f}%")
    print(f"  Mean reprice error: {df['Reprice_Error'].mean():.5f} Rs")
    return df

# ================================================================
# SECTION 4: PLOTS
# ================================================================

def plot_results(df, outfile='step1_bs_baseline_realdata.png'):
    df_p = df.dropna(subset=['IV_classical']).copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Step 1: Classical Black-Scholes Baseline\n'
                 'Real Nifty ATM Options', fontsize=13, fontweight='bold')

    # IV and RV over time
    ax = axes[0, 0]
    ax.plot(df_p['Date'], df_p['IV_classical']*100,
            color='steelblue', lw=1.2, label='Classical IV')
    ax.plot(df_p['Date'], df_p['RV_22']*100,
            color='darkorange', lw=1.0, alpha=0.8, label='RV (22d)')
    ax.set_title('IV vs Realized Volatility')
    ax.set_ylabel('Volatility (%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # IV vs RV scatter
    ax = axes[0, 1]
    v = df_p.dropna(subset=['RV_22'])
    ax.scatter(v['RV_22']*100, v['IV_classical']*100,
               alpha=0.4, s=15, color='steelblue')
    if len(v) > 2:
        sl, ic, rv, *_ = stats.linregress(v['RV_22'], v['IV_classical'])
        xr = np.linspace(v['RV_22'].min(), v['RV_22'].max(), 100)
        ax.plot(xr*100, (sl*xr+ic)*100, 'r--', lw=1.5,
                label=f'R²={rv**2:.3f}')
    ax.set_xlabel('Realized Vol (%)'); ax.set_ylabel('Classical IV (%)')
    ax.set_title('IV vs RV Scatter')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Moneyness distribution
    ax = axes[1, 0]
    ax.hist(df_p['Moneyness'], bins=40,
            color='teal', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', ls='--', label='ATM (=1.0)')
    ax.set_xlabel('Moneyness (S/K)'); ax.set_ylabel('Count')
    ax.set_title('ATM Quality — Moneyness Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    # IV distribution
    ax = axes[1, 1]
    ax.hist(df_p['IV_classical']*100, bins=40,
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(df_p['IV_classical'].mean()*100, color='red', ls='--',
               label=f"Mean={df_p['IV_classical'].mean()*100:.1f}%")
    ax.set_xlabel('Classical IV (%)'); ax.set_ylabel('Count')
    ax.set_title('IV Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()

# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 1: CLASSICAL BLACK-SCHOLES BASELINE")
    print("Real Nifty Options Data")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    df = load_data('nifty_atm_options_baseline.csv')

    print("\n[2/3] Extracting implied volatility...")
    df = extract_iv(df)

    print("\n[3/3] Saving results...")
    out_cols = ['Date', 'Expiry', 'Strike', 'Spot', 'C_market',
                'T_days', 'T_years', 'Moneyness', 'r', 'RV_22',
                'IV_call', 'IV_put', 'IV_classical',
                'C_reprice', 'Reprice_Error']
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv('nifty_atm_iv_classical.csv', index=False)
    print("  Saved: nifty_atm_iv_classical.csv")

    plot_results(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    dv = df.dropna(subset=['IV_classical'])
    print(f"  Days           : {len(dv)}")
    print(f"  Mean IV        : {dv['IV_classical'].mean()*100:.2f}%")
    print(f"  Std IV         : {dv['IV_classical'].std()*100:.2f}%")
    print(f"  Mean RV        : {dv['RV_22'].mean()*100:.2f}%")
    print(f"  IV-RV corr     : {dv['IV_classical'].corr(dv['RV_22']):.4f}")
    print(f"  Reprice error  : {dv['Reprice_Error'].mean():.6f} Rs")
    print("""
STEP 1 COMPLETE.
Next: run step2_pinn.py
""")
