"""
================================================================
STEP 3 — COMPARISON & VALIDATION
PINN IV vs Classical IV vs Realized Volatility
================================================================
This is the first results chapter of the thesis.

Replicates and extends the comparison methodology from
TH8745 (Panda, 2015) by adding PINN IV as a third series.

INPUT:
  nifty_pinn_iv_results.csv      (from step2_pinn_v4.py)
  nifty_atm_iv_classical.csv     (from step1_bs_baseline_realdata.py)

OUTPUT:
  step3_comparison_table.csv     (summary statistics table)
  step3_comparison_plots.png     (4-panel comparison plot)
  step3_pinn_vs_classical.png    (PINN vs Classical scatter)

WHAT THIS PRODUCES FOR YOUR THESIS:
  Table 1 — Descriptive statistics: Mean, Std, Min, Max
  Table 2 — Accuracy metrics: MAE, RMSE, Correlation vs RV
  Table 3 — PINN vs Classical agreement metrics
  Figure 1 — Time series: PINN IV, Classical IV, RV overlaid
  Figure 2 — Scatter: PINN IV vs Classical IV
  Figure 3 — Scatter: both IVs vs Realized Vol

RUN:
  python step3_comparison.py
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os


# ================================================================
# SECTION 1: LOAD DATA
# ================================================================

def load_data():
    """
    Loads PINN IV results and Classical IV results.
    Merges them on Date for comparison.
    """

    # ── PINN IV (one observation per 22-day window) ───────────
    pinn_file = 'nifty_pinn_iv_results.csv'
    if not os.path.exists(pinn_file):
        raise FileNotFoundError(
            f"'{pinn_file}' not found. Run step2_pinn_v4.py first.")
    df_pinn = pd.read_csv(pinn_file)
    df_pinn['Date'] = pd.to_datetime(df_pinn['Date'])
    print(f"  PINN IV: {len(df_pinn)} observations")
    print(f"  Columns: {df_pinn.columns.tolist()}")

    # ── Classical IV (one observation per trading day) ─────────
    cl_file = 'nifty_atm_iv_classical.csv'
    if not os.path.exists(cl_file):
        raise FileNotFoundError(
            f"'{cl_file}' not found. Run step1_bs_baseline_realdata.py first.")
    df_cl = pd.read_csv(cl_file)
    df_cl['Date'] = pd.to_datetime(df_cl['Date'])
    df_cl = df_cl.sort_values('Date').reset_index(drop=True)
    print(f"  Classical IV: {len(df_cl)} observations")

    # ── Match PINN windows to classical IV ────────────────────
    # PINN gives one IV per 22-day window (mid-window date).
    # For comparison we match each PINN window to the mean
    # classical IV and mean RV over that same window.

    records = []
    for _, row in df_pinn.iterrows():
        w_start = pd.to_datetime(row['Window_Start'])
        w_end   = pd.to_datetime(row['Window_End'])

        # Classical IV and RV within this window
        mask = ((df_cl['Date'] >= w_start) &
                (df_cl['Date'] <= w_end))
        win_cl = df_cl[mask]

        if len(win_cl) == 0:
            continue

        records.append({
            'Date'          : row['Date'],
            'Window_Start'  : w_start,
            'Window_End'    : w_end,
            'PINN_IV'       : row['PINN_IV'],
            'Classical_IV'  : win_cl['IV_classical'].mean(),
            'RV_22'         : win_cl['RV_22'].mean(),
            'r'             : win_cl['r'].mean(),
            'n_days'        : len(win_cl),
        })

    df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['PINN_IV', 'Classical_IV', 'RV_22'])

    print(f"\n  Matched windows: {len(df)}")
    print(f"  Date range: {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")

    return df, df_cl


# ================================================================
# SECTION 2: DESCRIPTIVE STATISTICS
# ================================================================

def descriptive_stats(df):
    """
    Table 1: Descriptive statistics for all three series.
    Matches Table format used in thesis Chapter 3.
    """
    series = {
        'PINN IV'       : df['PINN_IV']      * 100,
        'Classical IV'  : df['Classical_IV'] * 100,
        'Realized Vol'  : df['RV_22']        * 100,
    }

    rows = []
    for name, s in series.items():
        rows.append({
            'Series'   : name,
            'Mean (%)'  : round(s.mean(), 4),
            'Std (%)'   : round(s.std(), 4),
            'Min (%)'   : round(s.min(), 4),
            'Max (%)'   : round(s.max(), 4),
            'Median (%)': round(s.median(), 4),
            'Skewness'  : round(s.skew(), 4),
            'Kurtosis'  : round(s.kurtosis(), 4),
        })

    df_stats = pd.DataFrame(rows).set_index('Series')
    return df_stats


# ================================================================
# SECTION 3: ACCURACY METRICS VS REALIZED VOL
# ================================================================

def accuracy_metrics(df):
    """
    Table 2: How well does each IV measure predict RV?
    MAE, RMSE, Correlation, R-squared.
    These are the same metrics used in thesis Chapter 5.
    """
    rows = []
    for name, col in [('PINN IV', 'PINN_IV'),
                      ('Classical IV', 'Classical_IV')]:
        iv = df[col].values
        rv = df['RV_22'].values

        mae  = np.mean(np.abs(iv - rv))
        rmse = np.sqrt(np.mean((iv - rv)**2))
        corr = np.corrcoef(iv, rv)[0, 1]

        # OLS: RV = a + b*IV
        slope, intercept, r_val, p_val, se = stats.linregress(iv, rv)

        rows.append({
            'Series'        : name,
            'MAE (%)'       : round(mae * 100, 4),
            'RMSE (%)'      : round(rmse * 100, 4),
            'Correlation'   : round(corr, 4),
            'R-squared'     : round(r_val**2, 4),
            'Beta (IV->RV)' : round(slope, 4),
            'Alpha'         : round(intercept, 4),
            'p-value (beta)': round(p_val, 4),
        })

    df_acc = pd.DataFrame(rows).set_index('Series')
    return df_acc


# ================================================================
# SECTION 4: PINN vs CLASSICAL AGREEMENT
# ================================================================

def pinn_vs_classical(df):
    """
    Table 3: How closely does PINN IV track Classical IV?
    This validates the PINN extraction method.
    """
    diff = (df['PINN_IV'] - df['Classical_IV']) * 100
    corr = np.corrcoef(df['PINN_IV'], df['Classical_IV'])[0, 1]

    slope, intercept, r_val, p_val, _ = stats.linregress(
        df['Classical_IV'], df['PINN_IV'])

    metrics = {
        'Mean Difference (pp)'   : round(diff.mean(), 4),
        'Std of Difference (pp)' : round(diff.std(), 4),
        'Max Abs Difference (pp)': round(diff.abs().max(), 4),
        'MAE vs Classical (%)'   : round(diff.abs().mean(), 4),
        'RMSE vs Classical (%)'  : round(np.sqrt((diff**2).mean()), 4),
        'Correlation'            : round(corr, 4),
        'R-squared'              : round(r_val**2, 4),
        'Beta (Classical->PINN)' : round(slope, 4),
        'Alpha'                  : round(intercept, 4),
    }

    df_agree = pd.DataFrame.from_dict(
        metrics, orient='index', columns=['Value'])
    return df_agree


# ================================================================
# SECTION 5: PLOTS
# ================================================================

def plot_comparison(df, df_cl_daily,
                    outfile='step3_comparison_plots.png'):
    """
    4-panel comparison plot for thesis Figure 1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'Step 3: PINN IV vs Classical IV vs Realized Volatility\n'
        'Nifty 50 Index Options (2025–2026)',
        fontsize=13, fontweight='bold')

    # ── Panel 1: Time series (window-level) ──────────────────
    ax = axes[0, 0]
    ax.plot(df['Date'], df['PINN_IV'] * 100,
            'b-o', lw=1.8, ms=6, label='PINN IV', zorder=3)
    ax.plot(df['Date'], df['Classical_IV'] * 100,
            'g--s', lw=1.5, ms=5, label='Classical IV', zorder=2)
    ax.plot(df['Date'], df['RV_22'] * 100,
            color='darkorange', lw=1.2, alpha=0.8,
            marker='^', ms=4, label='Realized Vol (22d)')
    ax.set_title('Implied Volatility Time Series (Window Level)')
    ax.set_ylabel('Volatility (%)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # ── Panel 2: Daily classical IV vs daily RV ───────────────
    ax = axes[0, 1]
    df_cl_plot = df_cl_daily.dropna(
        subset=['IV_classical', 'RV_22'])
    ax.plot(df_cl_plot['Date'],
            df_cl_plot['IV_classical'] * 100,
            color='green', lw=1.0, alpha=0.8,
            label='Classical IV (daily)')
    ax.plot(df_cl_plot['Date'],
            df_cl_plot['RV_22'] * 100,
            color='darkorange', lw=1.0, alpha=0.6,
            label='Realized Vol (daily)')
    ax.set_title('Daily Classical IV vs Realized Vol')
    ax.set_ylabel('Volatility (%)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # ── Panel 3: PINN IV vs Classical IV scatter ──────────────
    ax = axes[1, 0]
    ax.scatter(df['Classical_IV'] * 100,
               df['PINN_IV'] * 100,
               color='steelblue', s=80, alpha=0.8, zorder=3)

    # Label each point with window number
    for i, row in df.iterrows():
        ax.annotate(f"W{i+1}",
                    (row['Classical_IV']*100, row['PINN_IV']*100),
                    fontsize=7, ha='left', va='bottom')

    # 45-degree line (perfect agreement)
    lims = [min(df['Classical_IV'].min(), df['PINN_IV'].min()) * 100 - 0.5,
            max(df['Classical_IV'].max(), df['PINN_IV'].max()) * 100 + 0.5]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect agreement')

    # OLS fit
    sl, ic, rv, pv, _ = stats.linregress(
        df['Classical_IV'], df['PINN_IV'])
    xr = np.linspace(df['Classical_IV'].min(),
                     df['Classical_IV'].max(), 100)
    ax.plot(xr * 100, (sl * xr + ic) * 100,
            'b-', lw=1.2,
            label=f'OLS: β={sl:.3f}, R²={rv**2:.3f}')

    ax.set_xlabel('Classical IV (%)')
    ax.set_ylabel('PINN IV (%)')
    ax.set_title('PINN IV vs Classical IV')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 4: Both IVs vs Realized Vol ─────────────────────
    ax = axes[1, 1]
    ax.scatter(df['RV_22'] * 100, df['PINN_IV'] * 100,
               color='steelblue', s=70, alpha=0.8,
               label='PINN IV', marker='o', zorder=3)
    ax.scatter(df['RV_22'] * 100, df['Classical_IV'] * 100,
               color='green', s=70, alpha=0.6,
               label='Classical IV', marker='s', zorder=2)

    # OLS lines
    for col, color, label in [
        ('PINN_IV', 'steelblue', 'PINN'),
        ('Classical_IV', 'green', 'Classical')
    ]:
        sl2, ic2, rv2, _, _ = stats.linregress(
            df['RV_22'], df[col])
        xr2 = np.linspace(df['RV_22'].min(),
                          df['RV_22'].max(), 100)
        ax.plot(xr2 * 100, (sl2 * xr2 + ic2) * 100,
                '-', color=color, lw=1.2,
                label=f'{label} R²={rv2**2:.3f}')

    ax.set_xlabel('Realized Vol (%)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('Both IVs vs Realized Vol')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


def plot_pinn_vs_classical(df,
        outfile='step3_pinn_vs_classical.png'):
    """
    Clean single scatter plot of PINN IV vs Classical IV.
    For thesis Figure 2.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(df['Classical_IV'] * 100,
                    df['PINN_IV'] * 100,
                    c=range(len(df)), cmap='viridis',
                    s=100, alpha=0.9, zorder=3)
    plt.colorbar(sc, ax=ax, label='Window number')

    # Perfect agreement line
    lims = [min(df['Classical_IV'].min(),
                df['PINN_IV'].min()) * 100 - 0.3,
            max(df['Classical_IV'].max(),
                df['PINN_IV'].max()) * 100 + 0.3]
    ax.plot(lims, lims, 'r--', lw=1.5,
            label='Perfect agreement (slope=1)')

    # OLS
    sl, ic, rv, pv, se = stats.linregress(
        df['Classical_IV'], df['PINN_IV'])
    xr = np.linspace(df['Classical_IV'].min(),
                     df['Classical_IV'].max(), 100)
    ax.plot(xr * 100, (sl * xr + ic) * 100,
            'b-', lw=1.5,
            label=f'OLS fit: β={sl:.4f}, α={ic*100:.4f}, '
                  f'R²={rv**2:.4f}')

    ax.set_xlabel('Classical IV (%)', fontsize=12)
    ax.set_ylabel('PINN IV (%)', fontsize=12)
    ax.set_title('PINN IV vs Classical IV\n'
                 'Nifty 50 ATM Options (2025–2026)',
                 fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


# ================================================================
# SECTION 6: PRINT TABLES
# ================================================================

def print_tables(df_stats, df_acc, df_agree):
    print("\n" + "=" * 65)
    print("TABLE 1: DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(df_stats.to_string())

    print("\n" + "=" * 65)
    print("TABLE 2: ACCURACY vs REALIZED VOLATILITY")
    print("=" * 65)
    print(df_acc.to_string())

    print("\n" + "=" * 65)
    print("TABLE 3: PINN IV vs CLASSICAL IV AGREEMENT")
    print("=" * 65)
    print(df_agree.to_string())


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STEP 3: COMPARISON & VALIDATION")
    print("PINN IV vs Classical IV vs Realized Volatility")
    print("=" * 65)

    # Load
    print("\n[1/4] Loading data...")
    df, df_cl_daily = load_data()

    # Tables
    print("\n[2/4] Computing statistics...")
    df_stats = descriptive_stats(df)
    df_acc   = accuracy_metrics(df)
    df_agree = pinn_vs_classical(df)
    print_tables(df_stats, df_acc, df_agree)

    # Save comparison table
    print("\n[3/4] Saving comparison table...")
    with open('step3_comparison_table.txt', 'w') as f:
        f.write("STEP 3: COMPARISON & VALIDATION\n")
        f.write("PINN IV vs Classical IV vs Realized Volatility\n")
        f.write("Nifty 50 ATM Options (2025-2026)\n\n")
        f.write("TABLE 1: DESCRIPTIVE STATISTICS\n")
        f.write(df_stats.to_string() + "\n\n")
        f.write("TABLE 2: ACCURACY vs REALIZED VOLATILITY\n")
        f.write(df_acc.to_string() + "\n\n")
        f.write("TABLE 3: PINN IV vs CLASSICAL IV AGREEMENT\n")
        f.write(df_agree.to_string() + "\n")
    print("  Saved: step3_comparison_table.txt")

    df_stats.to_csv('step3_descriptive_stats.csv')
    df_acc.to_csv('step3_accuracy_metrics.csv')
    df_agree.to_csv('step3_pinn_vs_classical_agreement.csv')

    # Plots
    print("\n[4/4] Generating plots...")
    plot_comparison(df, df_cl_daily,
                    'step3_comparison_plots.png')
    plot_pinn_vs_classical(df,
                    'step3_pinn_vs_classical.png')

    print("\n" + "=" * 65)
    print("STEP 3 COMPLETE")
    print("=" * 65)
    print(f"""
Files saved:
  step3_comparison_table.txt       <- thesis tables (copy directly)
  step3_descriptive_stats.csv      <- Table 1
  step3_accuracy_metrics.csv       <- Table 2
  step3_pinn_vs_classical_agreement.csv  <- Table 3
  step3_comparison_plots.png       <- 4-panel thesis figure
  step3_pinn_vs_classical.png      <- clean scatter thesis figure

Key findings to report in thesis:
  - Mean PINN IV vs Mean Classical IV (Table 1)
  - PINN-Classical correlation and R² (Table 3)
  - Both IVs correlation with RV (Table 2)
  - Whether PINN or Classical tracks RV better (Table 2, RMSE)

Next: run step4_information_content.py
""")