"""
================================================================
STEP 5 — PREDICTABILITY REGRESSIONS
Replicating and Extending Thesis Chapter 5
================================================================
The original thesis compared IV against four backward-looking
volatility measures as predictors of realized volatility:
  1. MA    — Simple Moving Average (22-day)
  2. EWMA  — Exponentially Weighted Moving Average (lambda=0.94)
  3. GARCH — GARCH(1,1)
  4. EGARCH — EGARCH(1,1)

This step adds two new competitors:
  5. Classical IV  (replicates thesis)
  6. PINN IV       (novel contribution)

For each forecasting method:
  IN-SAMPLE:  RV_{t+1} = alpha + beta * Forecast_t + epsilon
              Report R², beta, p-value, MZ unbiasedness test

  OUT-OF-SAMPLE: Split 70/30 train/test
              Forecast RV on test set using each method
              Report MSE, MAE, RMSE, MAPE
              These are the exact 4 metrics the thesis used

INPUT:
  nifty_pinn_iv_results.csv
  nifty_atm_iv_classical.csv
  nifty_spot_data.csv          (for MA, EWMA, GARCH, EGARCH)

OUTPUT:
  step5_predictability_results.txt   <- thesis Tables 5 and 6
  step5_predictability_plots.png     <- thesis Figure 5

RUN:
  pip install arch statsmodels
  python step5_predictability.py
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
from scipy import stats

# GARCH/EGARCH from arch library
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("  WARNING: arch library not found. "
          "GARCH/EGARCH will be skipped.")
    print("  Install with: pip install arch")


# ================================================================
# SECTION 1: LOAD DATA
# ================================================================

def load_data():
    """
    Loads all data sources and constructs the forecasting dataset.
    Window-level dataset with one observation per 22-day window.
    """
    # PINN results
    df_pinn = pd.read_csv('nifty_pinn_iv_results.csv')
    df_pinn['Date']         = pd.to_datetime(df_pinn['Date'])
    df_pinn['Window_Start'] = pd.to_datetime(df_pinn['Window_Start'])
    df_pinn['Window_End']   = pd.to_datetime(df_pinn['Window_End'])

    # Classical IV daily
    df_cl = pd.read_csv('nifty_atm_iv_classical.csv')
    df_cl['Date'] = pd.to_datetime(df_cl['Date'])
    df_cl = df_cl.sort_values('Date').reset_index(drop=True)

    # Spot data for MA, EWMA, GARCH, EGARCH
    df_spot = pd.read_csv('nifty_spot_data.csv')
    df_spot['Date'] = pd.to_datetime(df_spot['Date'], utc=True)
    df_spot['Date'] = df_spot['Date'].dt.tz_convert(
        'Asia/Kolkata').dt.date
    df_spot['Date'] = pd.to_datetime(df_spot['Date'])
    df_spot = df_spot.sort_values('Date').reset_index(drop=True)
    df_spot['Log_Return'] = np.log(
        df_spot['Close'] / df_spot['Close'].shift(1))
    df_spot = df_spot.dropna(subset=['Log_Return'])

    # Build window-level dataset
    records = []
    for _, row in df_pinn.iterrows():
        ws   = row['Window_Start']
        we   = row['Window_End']
        mask = (df_cl['Date'] >= ws) & (df_cl['Date'] <= we)
        win  = df_cl[mask]

        # Spot data for this window (for MA/EWMA)
        smask = (df_spot['Date'] >= ws) & (df_spot['Date'] <= we)
        swin  = df_spot[smask]

        if len(win) == 0 or len(swin) == 0:
            continue

        rv_current = win['RV_22'].mean()

        # MA: mean of daily realized vol over window
        ma_vol = win['RV_22'].mean()

        # EWMA: exponentially weighted vol (lambda=0.94, thesis)
        lam    = 0.94
        rets   = swin['Log_Return'].values
        ewma_var = rets[0]**2
        for r in rets[1:]:
            ewma_var = lam * ewma_var + (1 - lam) * r**2
        ewma_vol = np.sqrt(ewma_var * 252)

        records.append({
            'Date'        : row['Date'],
            'Window_Start': ws,
            'Window_End'  : we,
            'RV_current'  : rv_current,
            'PINN_IV'     : row['PINN_IV'],
            'Classical_IV': win['IV_classical'].mean(),
            'MA'          : ma_vol,
            'EWMA'        : ewma_vol,
        })

    df = pd.DataFrame(records).sort_values(
        'Date').reset_index(drop=True)

    # RV_{t+1} = next window's RV
    df['RV_next'] = df['RV_current'].shift(-1)
    df = df.dropna(subset=['RV_next'])

    print(f"  Dataset: {len(df)} observations")

    # ── GARCH(1,1) forecasts ──────────────────────────────────
    if ARCH_AVAILABLE:
        df = add_garch_forecasts(df, df_pinn, df_spot)
    else:
        df['GARCH']  = np.nan
        df['EGARCH'] = np.nan

    return df


def add_garch_forecasts(df, df_pinn, df_spot):
    """
    Fits GARCH(1,1) and EGARCH(1,1) on all spot returns
    up to each window and produces one-step-ahead forecasts.
    Exactly matches thesis methodology.
    """
    print("  Fitting GARCH/EGARCH models per window...")
    garch_forecasts  = []
    egarch_forecasts = []

    for _, row in df.iterrows():
        ws = pd.to_datetime(row['Window_Start'])

        # Use all returns up to start of this window
        hist = df_spot[df_spot['Date'] < ws]['Log_Return'].values * 100

        if len(hist) < 100:
            garch_forecasts.append(np.nan)
            egarch_forecasts.append(np.nan)
            continue

        try:
            # GARCH(1,1)
            gm = arch_model(hist, vol='Garch', p=1, q=1,
                           dist='normal', rescale=False)
            gr = gm.fit(disp='off', show_warning=False)
            gf = gr.forecast(horizon=22)
            # 22-day ahead annualized volatility
            garch_vol = np.sqrt(
                gf.variance.values[-1, :].mean() * 252) / 100
            garch_forecasts.append(garch_vol)
        except Exception:
            garch_forecasts.append(np.nan)

        try:
            # EGARCH(1,1)
            em = arch_model(hist, vol='EGARCH', p=1, q=1,
                           dist='normal', rescale=False)
            er = em.fit(disp='off', show_warning=False)
            ef = er.forecast(horizon=22)
            egarch_vol = np.sqrt(
                ef.variance.values[-1, :].mean() * 252) / 100
            egarch_forecasts.append(egarch_vol)
        except Exception:
            egarch_forecasts.append(np.nan)

    df['GARCH']  = garch_forecasts
    df['EGARCH'] = egarch_forecasts
    print(f"  GARCH fitted: "
          f"{sum(~np.isnan(df['GARCH']))}/{len(df)} windows")
    return df


# ================================================================
# SECTION 2: IN-SAMPLE REGRESSIONS
# ================================================================

def in_sample_regression(df, forecast_col, name):
    """
    In-sample: RV_{t+1} = alpha + beta * Forecast_t + epsilon
    Returns key metrics for the comparison table.
    """
    valid = df[[forecast_col, 'RV_next']].dropna()
    if len(valid) < 5:
        return {
            'Method': name, 'N': len(valid),
            'R2': np.nan, 'Adj_R2': np.nan,
            'Beta': np.nan, 'p_beta': np.nan,
            'MZ_p': np.nan, 'Unbiased': 'N/A'
        }

    y = valid['RV_next']
    X = sm.add_constant(valid[[forecast_col]])
    m = sm.OLS(y, X).fit(cov_type='HC3')

    # MZ test: H0 alpha=0, beta=1
    try:
        R = np.array([[1, 0], [0, 1]])
        r = np.array([0, 1])
        ft   = m.f_test((R, r))
        mz_p = float(ft.pvalue)
    except Exception:
        mz_p = np.nan

    return {
        'Method'   : name,
        'N'        : int(m.nobs),
        'R2'       : round(m.rsquared, 4),
        'Adj_R2'   : round(m.rsquared_adj, 4),
        'Beta'     : round(m.params[forecast_col], 4),
        'p_beta'   : round(m.pvalues[forecast_col], 4),
        'MZ_p'     : round(mz_p, 4) if not np.isnan(mz_p) else np.nan,
        'Unbiased' : ('Yes' if (not np.isnan(mz_p) and mz_p > 0.05)
                      else 'No')
    }


# ================================================================
# SECTION 3: OUT-OF-SAMPLE FORECASTING
# ================================================================

def out_of_sample_metrics(df, forecast_col, name,
                          train_ratio=0.70):
    """
    Out-of-sample: split 70/30 train/test.
    Fit regression on train set.
    Forecast RV on test set.
    Compute MSE, MAE, RMSE, MAPE — exact 4 metrics from thesis.
    """
    valid = df[[forecast_col, 'RV_next']].dropna().reset_index(drop=True)
    n     = len(valid)

    if n < 6:
        return {
            'Method': name,
            'MSE': np.nan, 'MAE': np.nan,
            'RMSE': np.nan, 'MAPE': np.nan,
            'N_test': 0
        }

    split  = int(np.floor(n * train_ratio))
    train  = valid.iloc[:split]
    test   = valid.iloc[split:]

    if len(train) < 4 or len(test) < 2:
        return {
            'Method': name,
            'MSE': np.nan, 'MAE': np.nan,
            'RMSE': np.nan, 'MAPE': np.nan,
            'N_test': len(test)
        }

    # Fit on train
    y_tr = train['RV_next']
    X_tr = sm.add_constant(train[[forecast_col]])
    m    = sm.OLS(y_tr, X_tr).fit()

    # Forecast on test
    X_te     = sm.add_constant(test[[forecast_col]])
    y_pred   = m.predict(X_te)
    y_actual = test['RV_next'].values
    y_pred   = y_pred.values

    mse  = np.mean((y_actual - y_pred)**2)
    mae  = np.mean(np.abs(y_actual - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(
        (y_actual - y_pred) / (y_actual + 1e-10))) * 100

    return {
        'Method' : name,
        'N_test' : len(test),
        'MSE'    : round(mse * 10000, 6),   # in (%)^2
        'MAE'    : round(mae * 100,   4),    # in %
        'RMSE'   : round(rmse * 100,  4),    # in %
        'MAPE'   : round(mape,        4),    # in %
    }


# ================================================================
# SECTION 4: PLOTS
# ================================================================

def plot_predictability(df, methods, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'Step 5: Predictability Regressions\n'
        'Forecasting Realized Volatility — In-Sample & Out-of-Sample',
        fontsize=13, fontweight='bold')

    colors = {
        'MA': 'gray', 'EWMA': 'purple',
        'GARCH': 'brown', 'EGARCH': 'olive',
        'Classical IV': 'green', 'PINN IV': 'steelblue'
    }

    # ── Panel 1: In-sample R² comparison ──────────────────────
    ax = axes[0, 0]
    is_data = [(m['Method'], m['R2'])
               for m in methods['in_sample']
               if not np.isnan(m['R2'])]
    if is_data:
        names_is, r2_is = zip(*is_data)
        clrs = [colors.get(n, 'steelblue') for n in names_is]
        bars = ax.bar(names_is, [v*100 for v in r2_is],
                      color=clrs, alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, r2_is):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.3f}', ha='center',
                    fontsize=9, fontweight='bold')
    ax.set_title('In-Sample R² by Method')
    ax.set_ylabel('R² (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Out-of-sample RMSE comparison ────────────────
    ax = axes[0, 1]
    oos_data = [(m['Method'], m['RMSE'])
                for m in methods['out_of_sample']
                if not np.isnan(m['RMSE'])]
    if oos_data:
        names_oos, rmse_oos = zip(*oos_data)
        clrs2 = [colors.get(n, 'steelblue') for n in names_oos]
        bars2 = ax.bar(names_oos, rmse_oos,
                       color=clrs2, alpha=0.8, edgecolor='white')
        for bar, val in zip(bars2, rmse_oos):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center',
                    fontsize=9, fontweight='bold')
    ax.set_title('Out-of-Sample RMSE by Method (lower = better)')
    ax.set_ylabel('RMSE (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Forecasts vs actual RV over time ─────────────
    ax = axes[1, 0]
    ax.plot(df['Date'], df['RV_next'] * 100,
            'k-', lw=2.0, label='Actual RV', zorder=5)
    for col, label, color in [
        ('PINN_IV',     'PINN IV',     'steelblue'),
        ('Classical_IV','Classical IV','green'),
        ('MA',          'MA',          'gray'),
        ('EWMA',        'EWMA',        'purple'),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.plot(df['Date'], df[col] * 100,
                    '-', color=color, lw=1.0, alpha=0.7,
                    label=label)
    ax.set_title('All Forecasts vs Actual RV')
    ax.set_ylabel('Volatility (%)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # ── Panel 4: PINN vs Classical forecast scatter ───────────
    ax = axes[1, 1]
    ax.scatter(df['PINN_IV'] * 100, df['RV_next'] * 100,
               color='steelblue', s=70, alpha=0.8,
               label='PINN IV', zorder=3)
    ax.scatter(df['Classical_IV'] * 100, df['RV_next'] * 100,
               color='green', s=70, alpha=0.6, marker='s',
               label='Classical IV', zorder=2)

    # Reference line
    all_vals = pd.concat([
        df['PINN_IV'], df['Classical_IV'],
        df['RV_next']]).dropna() * 100
    lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
    ax.plot(lims, lims, 'r--', lw=1.2, alpha=0.5,
            label='Perfect forecast')
    ax.set_xlabel('Forecast (%)')
    ax.set_ylabel('Actual RV_{t+1} (%)')
    ax.set_title('PINN vs Classical: Forecast Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outfile}")
    plt.close()


# ================================================================
# SECTION 5: PRINT AND SAVE TABLES
# ================================================================

def print_tables(is_rows, oos_rows):
    print("\n" + "=" * 75)
    print("TABLE 5: IN-SAMPLE REGRESSION RESULTS")
    print("RV_{t+1} = alpha + beta * Forecast_t + epsilon")
    print("=" * 75)
    print(f"{'Method':<16} {'N':>4} {'R2':>7} {'Adj R2':>8} "
          f"{'Beta':>8} {'p(beta)':>9} {'MZ p':>8} {'Unbiased':>10}")
    print("-" * 75)
    for r in is_rows:
        sig = ('***' if (not np.isnan(r['p_beta']) and r['p_beta'] < 0.01)
               else '**' if (not np.isnan(r['p_beta']) and r['p_beta'] < 0.05)
               else '*'  if (not np.isnan(r['p_beta']) and r['p_beta'] < 0.10)
               else '')
        r2  = f"{r['R2']:.4f}"    if not np.isnan(r['R2'])    else 'N/A'
        ar2 = f"{r['Adj_R2']:.4f}" if not np.isnan(r['Adj_R2']) else 'N/A'
        bt  = f"{r['Beta']:.4f}"  if not np.isnan(r['Beta'])  else 'N/A'
        pb  = f"{r['p_beta']:.4f}{sig}" if not np.isnan(r['p_beta']) else 'N/A'
        mz  = f"{r['MZ_p']:.4f}"  if not np.isnan(r['MZ_p'])  else 'N/A'
        print(f"  {r['Method']:<14} {r['N']:>4} {r2:>7} {ar2:>8} "
              f"{bt:>8} {pb:>9} {mz:>8} {r['Unbiased']:>10}")

    print("\n*** p<0.01  ** p<0.05  * p<0.10")

    print("\n" + "=" * 65)
    print("TABLE 6: OUT-OF-SAMPLE FORECAST ERRORS (70/30 split)")
    print("= " * 32)
    print(f"{'Method':<16} {'N_test':>7} {'MSE(x10^4)':>12} "
          f"{'MAE(%)':>8} {'RMSE(%)':>9} {'MAPE(%)':>9}")
    print("-" * 65)
    for r in oos_rows:
        mse  = f"{r['MSE']:.4f}"  if not np.isnan(r['MSE'])  else 'N/A'
        mae  = f"{r['MAE']:.4f}"  if not np.isnan(r['MAE'])  else 'N/A'
        rmse = f"{r['RMSE']:.4f}" if not np.isnan(r['RMSE']) else 'N/A'
        mape = f"{r['MAPE']:.4f}" if not np.isnan(r['MAPE']) else 'N/A'
        print(f"  {r['Method']:<14} {r['N_test']:>7} {mse:>12} "
              f"{mae:>8} {rmse:>9} {mape:>9}")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STEP 5: PREDICTABILITY REGRESSIONS")
    print("Replicating and Extending Thesis Chapter 5")
    print("=" * 65)

    print("\n[1/3] Loading data...")
    df = load_data()

    print(f"\n  Columns: {df.columns.tolist()}")
    print(f"\n  Sample means (forecasting variables):")
    for col in ['MA','EWMA','GARCH','EGARCH',
                'Classical_IV','PINN_IV','RV_next']:
        if col in df.columns:
            mn = df[col].mean()
            print(f"    {col:<15}: {mn*100:.2f}%")

    print("\n[2/3] Running regressions...")

    # Define all forecasting methods
    forecast_methods = [
        ('MA',           'MA'),
        ('EWMA',         'EWMA'),
        ('GARCH',        'GARCH'),
        ('EGARCH',       'EGARCH'),
        ('Classical_IV', 'Classical IV'),
        ('PINN_IV',      'PINN IV'),
    ]

    # In-sample
    is_rows  = []
    oos_rows = []
    for col, name in forecast_methods:
        if col not in df.columns:
            continue
        is_rows.append(in_sample_regression(df, col, name))
        oos_rows.append(out_of_sample_metrics(df, col, name))

    print_tables(is_rows, oos_rows)

    print("\n[3/3] Saving results and plots...")

    # Save tables
    with open('step5_predictability_results.txt', 'w',
              encoding='utf-8') as f:
        f.write("STEP 5: PREDICTABILITY REGRESSIONS\n")
        f.write("Nifty 50 ATM Options (2025-2026)\n\n")
        f.write("TABLE 5: IN-SAMPLE REGRESSION RESULTS\n")
        f.write(f"{'Method':<16} {'N':>4} {'R2':>7} "
                f"{'Beta':>8} {'p(beta)':>9} {'Unbiased':>10}\n")
        for r in is_rows:
            r2 = f"{r['R2']:.4f}" if not np.isnan(r['R2']) else 'N/A'
            bt = f"{r['Beta']:.4f}" if not np.isnan(r['Beta']) else 'N/A'
            pb = f"{r['p_beta']:.4f}" if not np.isnan(r['p_beta']) else 'N/A'
            f.write(f"  {r['Method']:<14} {r['N']:>4} {r2:>7} "
                    f"{bt:>8} {pb:>9} {r['Unbiased']:>10}\n")
        f.write("\nTABLE 6: OUT-OF-SAMPLE FORECAST ERRORS\n")
        f.write(f"{'Method':<16} {'MSE':>10} {'MAE':>8} "
                f"{'RMSE':>9} {'MAPE':>9}\n")
        for r in oos_rows:
            mse  = f"{r['MSE']:.4f}"  if not np.isnan(r['MSE'])  else 'N/A'
            mae  = f"{r['MAE']:.4f}"  if not np.isnan(r['MAE'])  else 'N/A'
            rmse = f"{r['RMSE']:.4f}" if not np.isnan(r['RMSE']) else 'N/A'
            mape = f"{r['MAPE']:.4f}" if not np.isnan(r['MAPE']) else 'N/A'
            f.write(f"  {r['Method']:<14} {mse:>10} {mae:>8} "
                    f"{rmse:>9} {mape:>9}\n")

    print("  Saved: step5_predictability_results.txt")

    plot_predictability(
        df,
        {'in_sample': is_rows, 'out_of_sample': oos_rows},
        'step5_predictability_plots.png')

    print("\n" + "=" * 65)
    print("STEP 5 COMPLETE")
    print("=" * 65)
    print("""
Files saved:
  step5_predictability_results.txt   <- thesis Tables 5 and 6
  step5_predictability_plots.png     <- thesis Figure 5

Key things to look for:
  TABLE 5 (in-sample):
    - Which method has highest R2?
    - Is PINN IV R2 >= Classical IV R2?
    - Which methods are unbiased (MZ test)?

  TABLE 6 (out-of-sample):
    - Which method has lowest RMSE? (best forecaster)
    - Is PINN IV RMSE <= Classical IV RMSE?
    - How do both IVs compare to GARCH/EGARCH?

  The thesis found IV outperformed all backward-looking
  measures out-of-sample. If PINN IV also outperforms,
  that strengthens the thesis contribution.

Note: With only 12 observations, out-of-sample split gives
~8 train / ~4 test observations. Results are directional.
More data years will make these results more robust.

Next: Step 6 (Volatility Surface) or Step 7 (Thesis Writing)
""")