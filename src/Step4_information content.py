"""
================================================================
STEP 4 — INFORMATION CONTENT REGRESSIONS
Replicating and Extending Thesis Chapter 4
================================================================
The original thesis (TH8745, Panda 2015) ran this regression:

    RV_{t+1} = α + β·IV_t + ε

to test whether implied volatility contains information about
future realized volatility. A significant β means IV is
informative. If β=1 and α=0, IV is also unbiased.

This step replicates that regression with:
  1. Classical IV  (baseline — replicates thesis)
  2. PINN IV       (novel contribution)
  3. Both together (multiple regression — tests if PINN adds
                    information beyond classical IV)

Also runs the Mincer-Zarnowitz (MZ) unbiasedness test:
    H0: α=0 AND β=1 simultaneously
    Rejection means IV is biased as a predictor of RV.

INPUT:
  nifty_pinn_iv_results.csv
  nifty_atm_iv_classical.csv

OUTPUT:
  step4_regression_results.txt    <- thesis Table 4
  step4_information_content.png   <- thesis Figure 4

RUN:
  pip install statsmodels
  python step4_information_content.py
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

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats


# ================================================================
# SECTION 1: LOAD AND PREPARE DATA
# ================================================================

def load_and_prepare():
    """
    Loads PINN IV and Classical IV.
    Constructs the regression dataset:
      IV_t  = implied volatility in window t
      RV_{t+1} = realized volatility in window t+1

    This is the exact specification from thesis Chapter 4.
    IV at time t should predict RV at time t+1.
    """
    # PINN results
    df_pinn = pd.read_csv('nifty_pinn_iv_results.csv')
    df_pinn['Date']         = pd.to_datetime(df_pinn['Date'])
    df_pinn['Window_Start'] = pd.to_datetime(df_pinn['Window_Start'])
    df_pinn['Window_End']   = pd.to_datetime(df_pinn['Window_End'])

    # Classical IV (daily)
    df_cl = pd.read_csv('nifty_atm_iv_classical.csv')
    df_cl['Date'] = pd.to_datetime(df_cl['Date'])
    df_cl = df_cl.sort_values('Date').reset_index(drop=True)

    # Build window-level dataset
    # For each PINN window, compute mean classical IV and mean RV
    records = []
    for _, row in df_pinn.iterrows():
        ws = row['Window_Start']
        we = row['Window_End']
        mask = (df_cl['Date'] >= ws) & (df_cl['Date'] <= we)
        win  = df_cl[mask]
        if len(win) == 0:
            continue
        records.append({
            'Date'        : row['Date'],
            'Window_Start': ws,
            'Window_End'  : we,
            'PINN_IV'     : row['PINN_IV'],
            'Classical_IV': win['IV_classical'].mean(),
            'RV_current'  : win['RV_22'].mean(),
        })

    df = pd.DataFrame(records).sort_values(
        'Date').reset_index(drop=True)

    # RV_{t+1} = next window's RV
    df['RV_next'] = df['RV_current'].shift(-1)

    # Drop last row (no future RV available)
    df = df.dropna(subset=['RV_next', 'PINN_IV', 'Classical_IV'])

    print(f"  Regression dataset: {len(df)} observations")
    print(f"  Date range: {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"\n  RV_next  : mean={df['RV_next'].mean()*100:.2f}%  "
          f"std={df['RV_next'].std()*100:.2f}%")
    print(f"  PINN IV  : mean={df['PINN_IV'].mean()*100:.2f}%  "
          f"std={df['PINN_IV'].std()*100:.2f}%")
    print(f"  Classic  : mean={df['Classical_IV'].mean()*100:.2f}%  "
          f"std={df['Classical_IV'].std()*100:.2f}%")

    return df


# ================================================================
# SECTION 2: RUN REGRESSIONS
# ================================================================

def run_regression(y, X, name):
    """
    OLS regression with full diagnostics.
    y: dependent variable (RV_{t+1})
    X: independent variables (IV_t or multiple IVs)
    name: label for printing
    """
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit(cov_type='HC3')
    # HC3 = heteroskedasticity-robust standard errors
    # More conservative than regular OLS, appropriate for
    # small samples like ours (12 observations)

    return model


def mincer_zarnowitz_test(model, y, X):
    """
    Mincer-Zarnowitz unbiasedness test.
    H0: α=0 AND β=1 simultaneously.

    Uses F-test on joint restriction:
      R * β = r
    where R = [[1,0],[0,1]] and r = [0,1]

    Rejection means IV is a biased predictor of RV.
    This was a key test in the original thesis.
    """
    # Joint hypothesis: const=0, slope=1
    X_const = sm.add_constant(X)
    n_vars  = X_const.shape[1]

    # Build restriction matrix
    # For simple regression (intercept + 1 slope):
    #   R = [[1, 0],   r = [0]   (alpha = 0)
    #        [0, 1]]       [1]   (beta  = 1)
    if n_vars == 2:
        R = np.array([[1, 0],
                      [0, 1]])
        r = np.array([0, 1])
    else:
        # Multiple regression — test first slope only
        R = np.array([[1, 0, 0],
                      [0, 1, 0]])
        r = np.array([0, 1])

    try:
        f_test = model.f_test((R, r))
        return {
            'F-stat' : round(float(f_test.fvalue), 4),
            'p-value': round(float(f_test.pvalue), 4),
            'Result' : 'Reject H0 (biased)' if float(f_test.pvalue) < 0.05
                       else 'Fail to reject H0 (unbiased)'
        }
    except Exception:
        return {'F-stat': np.nan, 'p-value': np.nan,
                'Result': 'Test failed'}


def format_regression_table(model, name, mz_test):
    """
    Formats regression output as a clean table for thesis.
    """
    params = model.params
    pvals  = model.pvalues
    tvals  = model.tvalues
    ci     = model.conf_int()

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"REGRESSION: {name}")
    lines.append(f"{'='*60}")
    lines.append(f"Dependent variable: RV_(t+1)")
    lines.append(f"Observations      : {int(model.nobs)}")
    lines.append(f"R-squared         : {model.rsquared:.4f}")
    lines.append(f"Adj. R-squared    : {model.rsquared_adj:.4f}")
    lines.append(f"F-statistic       : {model.fvalue:.4f} "
                 f"(p={model.f_pvalue:.4f})")
    lines.append(f"AIC               : {model.aic:.4f}")
    lines.append(f"Durbin-Watson     : "
                 f"{durbin_watson(model.resid):.4f}")
    lines.append(f"\nCoefficients (HC3 robust SEs):")
    lines.append(f"{'Variable':<20} {'Coef':>8} {'t-stat':>8} "
                 f"{'p-value':>8} {'95% CI'}")
    lines.append("-" * 60)

    var_names = ['const'] + [c for c in model.model.exog_names
                             if c != 'const']
    for vn in model.model.exog_names:
        sig = ('***' if pvals[vn] < 0.01 else
               '**'  if pvals[vn] < 0.05 else
               '*'   if pvals[vn] < 0.10 else '')
        lines.append(
            f"  {vn:<18} {params[vn]:>8.4f} "
            f"{tvals[vn]:>8.3f} "
            f"{pvals[vn]:>8.4f}{sig}  "
            f"[{ci.loc[vn,0]:.4f}, {ci.loc[vn,1]:.4f}]")

    lines.append(f"\nMincer-Zarnowitz Test (H0: α=0, β=1):")
    lines.append(f"  F-stat : {mz_test['F-stat']}")
    lines.append(f"  p-value: {mz_test['p-value']}")
    lines.append(f"  Result : {mz_test['Result']}")
    lines.append("*** p<0.01  ** p<0.05  * p<0.10")

    return '\n'.join(lines)


# ================================================================
# SECTION 3: PLOTS
# ================================================================

def plot_regressions(df, models, outfile='step4_information_content.png'):
    """
    4-panel plot showing regression fits and diagnostics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        'Step 4: Information Content Regressions\n'
        'RV$_{t+1}$ = α + β·IV$_t$ + ε',
        fontsize=13, fontweight='bold')

    rv_next = df['RV_next'].values * 100

    # ── Panel 1: PINN IV regression fit ──────────────────────
    ax = axes[0, 0]
    pinn_iv = df['PINN_IV'].values * 100
    m = models['PINN']
    fitted = m.fittedvalues.values * 100

    ax.scatter(pinn_iv, rv_next,
               color='steelblue', s=70, alpha=0.8, zorder=3,
               label='Observations')
    xr = np.linspace(pinn_iv.min(), pinn_iv.max(), 100)
    a, b = m.params['const'], m.params['PINN_IV']
    ax.plot(xr, a*100 + b*xr,
            'b-', lw=1.8,
            label=f'OLS: β={b:.3f}, R²={m.rsquared:.3f}')
    ax.plot([pinn_iv.min(), pinn_iv.max()],
            [pinn_iv.min(), pinn_iv.max()],
            'r--', lw=1.2, alpha=0.5, label='β=1 line')
    ax.set_xlabel('PINN IV$_t$ (%)')
    ax.set_ylabel('RV$_{t+1}$ (%)')
    ax.set_title('PINN IV → Future RV')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 2: Classical IV regression fit ──────────────────
    ax = axes[0, 1]
    cl_iv = df['Classical_IV'].values * 100
    m2    = models['Classical']
    a2, b2 = m2.params['const'], m2.params['Classical_IV']

    ax.scatter(cl_iv, rv_next,
               color='green', s=70, alpha=0.8, zorder=3,
               label='Observations')
    xr2 = np.linspace(cl_iv.min(), cl_iv.max(), 100)
    ax.plot(xr2, a2*100 + b2*xr2,
            'g-', lw=1.8,
            label=f'OLS: β={b2:.3f}, R²={m2.rsquared:.3f}')
    ax.plot([cl_iv.min(), cl_iv.max()],
            [cl_iv.min(), cl_iv.max()],
            'r--', lw=1.2, alpha=0.5, label='β=1 line')
    ax.set_xlabel('Classical IV$_t$ (%)')
    ax.set_ylabel('RV$_{t+1}$ (%)')
    ax.set_title('Classical IV → Future RV')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 3: Residuals over time ──────────────────────────
    ax = axes[1, 0]
    ax.plot(df['Date'], models['PINN'].resid * 100,
            'b-o', ms=5, lw=1.2, label='PINN residuals')
    ax.plot(df['Date'], models['Classical'].resid * 100,
            'g--s', ms=5, lw=1.2, label='Classical residuals')
    ax.axhline(0, color='black', lw=1.0, ls='-')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (%)')
    ax.set_title('Regression Residuals Over Time')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # ── Panel 4: R² comparison bar chart ──────────────────────
    ax = axes[1, 1]
    names  = ['PINN IV', 'Classical IV', 'Both IVs\n(Multiple)']
    r2vals = [models['PINN'].rsquared,
              models['Classical'].rsquared,
              models['Both'].rsquared]
    colors = ['steelblue', 'green', 'darkorange']
    bars   = ax.bar(names, [v * 100 for v in r2vals],
                    color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, r2vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.3f}', ha='center', fontsize=10,
                fontweight='bold')
    ax.set_ylabel('R² (%)')
    ax.set_title('R² Comparison Across Specifications')
    ax.set_ylim(0, max(r2vals)*100*1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STEP 4: INFORMATION CONTENT REGRESSIONS")
    print("Replicating and Extending Thesis Chapter 4")
    print("=" * 65)

    # Load data
    print("\n[1/3] Loading and preparing data...")
    df = load_and_prepare()

    # Run regressions
    print("\n[2/3] Running regressions...")

    y = df['RV_next']

    # Regression 1: PINN IV only
    X_pinn = df[['PINN_IV']]
    m_pinn = run_regression(y, X_pinn, 'PINN IV')
    mz_pinn = mincer_zarnowitz_test(m_pinn, y, X_pinn)

    # Regression 2: Classical IV only
    X_cl   = df[['Classical_IV']]
    m_cl   = run_regression(y, X_cl, 'Classical IV')
    mz_cl  = mincer_zarnowitz_test(m_cl, y, X_cl)

    # Regression 3: Both IVs (multiple regression)
    # Tests whether PINN IV adds information beyond Classical IV
    # If β_PINN is significant while β_Classical is not,
    # PINN IV subsumes classical IV
    X_both = df[['PINN_IV', 'Classical_IV']]
    m_both = run_regression(y, X_both, 'Both IVs')
    mz_both = mincer_zarnowitz_test(m_both, y, X_both)

    models = {
        'PINN'     : m_pinn,
        'Classical': m_cl,
        'Both'     : m_both,
    }

    # Print results
    t1 = format_regression_table(m_pinn,     'PINN IV Only',      mz_pinn)
    t2 = format_regression_table(m_cl,       'Classical IV Only', mz_cl)
    t3 = format_regression_table(m_both,     'Both IVs (Multiple Regression)', mz_both)

    print(t1)
    print(t2)
    print(t3)

    # Summary comparison table
    print("\n" + "=" * 65)
    print("SUMMARY: INFORMATION CONTENT COMPARISON")
    print("=" * 65)
    print(f"{'Specification':<30} {'R²':>6} {'Adj R²':>8} "
          f"{'β':>8} {'p(β)':>8} {'MZ p-val':>10}")
    print("-" * 65)
    specs = [
        ('PINN IV only',          m_pinn, 'PINN_IV'),
        ('Classical IV only',     m_cl,   'Classical_IV'),
    ]
    for name, m, bvar in specs:
        print(f"  {name:<28} {m.rsquared:>6.4f} "
              f"{m.rsquared_adj:>8.4f} "
              f"{m.params[bvar]:>8.4f} "
              f"{m.pvalues[bvar]:>8.4f} "
              f"{'***' if m.pvalues[bvar]<0.01 else '**' if m.pvalues[bvar]<0.05 else '*' if m.pvalues[bvar]<0.10 else ''}")

    print(f"\n  {'Both IVs (multiple)':<28} "
          f"{m_both.rsquared:>6.4f} "
          f"{m_both.rsquared_adj:>8.4f}")
    print(f"    β_PINN     = {m_both.params['PINN_IV']:.4f}  "
          f"p={m_both.pvalues['PINN_IV']:.4f}")
    print(f"    β_Classical= {m_both.params['Classical_IV']:.4f}  "
          f"p={m_both.pvalues['Classical_IV']:.4f}")

    # Save results
    print("\n[3/3] Saving results and plots...")
    with open('step4_regression_results.txt', 'w', encoding='utf-8') as f:
        f.write("STEP 4: INFORMATION CONTENT REGRESSIONS\n")
        f.write("RV_{t+1} = alpha + beta * IV_t + epsilon\n")
        f.write("Nifty 50 ATM Options (2025-2026)\n")
        f.write(t1 + "\n")
        f.write(t2 + "\n")
        f.write(t3 + "\n")
    print("  Saved: step4_regression_results.txt")

    plot_regressions(df, models, 'step4_information_content.png')

    print("\n" + "=" * 65)
    print("STEP 4 COMPLETE")
    print("=" * 65)
    print("""
Files saved:
  step4_regression_results.txt     <- full regression tables
  step4_information_content.png    <- 4-panel regression plot

Key things to look for in results:
  1. Is β significant? (p < 0.05 means IV is informative)
  2. Is R² higher for PINN or Classical? (higher = more info)
  3. In multiple regression, is β_PINN significant when
     β_Classical is also included? (yes = PINN adds info)
  4. MZ test: fail to reject means IV is unbiased predictor

Note on sample size:
  With only 12 observations (13 windows minus 1 for lead),
  results are indicative not conclusive. More years of data
  will give more power to these tests. Report this limitation
  in your thesis and note results will strengthen with more data.

Next: run step5_predictability.py
""")