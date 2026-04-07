"""
================================================================
STEP 0 — DATA CLEANING & ATM FILTERING
Real Nifty Options Data (NSE Historical Contract-wise)
================================================================
INPUT FILES (same folder as this script):
  NIFTY_CALL_OPTIONS_2025_2026.csv
  NIFTY_PUT_OPTIONS_2025_2026.csv
  nifty_spot_data.csv

OUTPUT FILES:
  nifty_atm_calls_clean.csv
  nifty_atm_puts_clean.csv
  nifty_atm_options_baseline.csv   <- feeds Step 1 and Step 2

RUN:
  python step0_data_cleaning.py
================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1: CONFIGURATION
# ================================================================

CALL_FILES = [
    'NIFTY_CALL_OPTIONS_2025_2026.csv',
    # Add more years below as you download them:
    # 'NIFTY_CALL_OPTIONS_2024_2025.csv',
    # 'NIFTY_CALL_OPTIONS_2023_2024.csv',
]

PUT_FILES = [
    'NIFTY_PUT_OPTIONS_2025_2026.csv',
    # 'NIFTY_PUT_OPTIONS_2024_2025.csv',
    # 'NIFTY_PUT_OPTIONS_2023_2024.csv',
]

SPOT_FILE = 'nifty_spot_data.csv'

MIN_DAYS_TO_EXPIRY = 6    # thesis: drop options expiring in <= 6 days
MAX_DAYS_TO_EXPIRY = 35   # near-month only
MIN_CLOSE_PRICE    = 0.5  # drop illiquid options worth < 0.5 Rs

# ================================================================
# SECTION 2: RISK-FREE RATE
# ================================================================

def get_risk_free_rate(date):
    """Annualized RBI repo rate approximation by year."""
    y = date.year
    if   y <= 2012: return 0.085
    elif y <= 2013: return 0.080
    elif y <= 2014: return 0.080
    elif y <= 2015: return 0.075
    elif y <= 2016: return 0.065
    elif y <= 2017: return 0.060
    elif y <= 2018: return 0.060
    elif y <= 2019: return 0.055
    elif y <= 2020: return 0.045
    elif y <= 2021: return 0.040
    elif y <= 2022: return 0.050
    elif y <= 2023: return 0.065
    elif y <= 2024: return 0.065
    else:           return 0.065

# ================================================================
# SECTION 3: LOAD AND CLEAN
# ================================================================

def load_options(file_list, label):
    dfs = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"  WARNING: File not found — {f}")
            continue
        tmp = pd.read_csv(f)
        dfs.append(tmp)
        print(f"  Loaded {f}: {len(tmp):,} rows")

    if not dfs:
        raise FileNotFoundError(f"No {label} files found. Check paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total {label} rows: {len(df):,}")

    # Strip whitespace from column names (NSE adds trailing spaces)
    df.columns = df.columns.str.strip()

    # Rename to clean names
    rename_map = {
        'Symbol'                          : 'Symbol',
        'Date'                            : 'Date',
        'Expiry'                          : 'Expiry',
        'Option type'                     : 'Option_Type',
        'Strike Price'                    : 'Strike',
        'Open'                            : 'Open',
        'High'                            : 'High',
        'Low'                             : 'Low',
        'Close'                           : 'Close',
        'LTP'                             : 'LTP',
        'Settle Price'                    : 'Settle_Price',
        'No. of contracts'                : 'Volume',
        'Turnover * in  \u20b9 Lakhs'     : 'Turnover',
        'Premium Turnover ** in   \u20b9 Lakhs': 'Premium_Turnover',
        'Open Int'                        : 'OI',
        'Change in OI'                    : 'Change_OI',
        'Underlying Value'                : 'Spot',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})

    # Keep only needed columns
    keep = ['Date', 'Expiry', 'Option_Type', 'Strike',
            'Close', 'Settle_Price', 'Spot']
    df = df[[c for c in keep if c in df.columns]]

    # Parse NSE dates: format is 11-Mar-2026
    df['Date']   = pd.to_datetime(
        df['Date'].astype(str).str.strip(),
        format='%d-%b-%Y', dayfirst=True)
    df['Expiry'] = pd.to_datetime(
        df['Expiry'].astype(str).str.strip(),
        format='%d-%b-%Y', dayfirst=True)

    # Convert dash (-) to NaN
    for col in ['Close', 'Settle_Price', 'Spot', 'Strike']:
        df[col] = df[col].astype(str).str.strip().replace('-', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fallback: use Settle_Price when Close is missing
    df['Close'] = df['Close'].fillna(df['Settle_Price'])

    # Drop rows missing essentials
    df = df.dropna(subset=['Date', 'Expiry', 'Strike', 'Close', 'Spot'])

    # Drop very cheap / illiquid options
    df = df[df['Close'] >= MIN_CLOSE_PRICE]

    # Days to expiry
    df['T_days'] = (df['Expiry'] - df['Date']).dt.days

    # Near-month filter (matches thesis)
    df = df[(df['T_days'] > MIN_DAYS_TO_EXPIRY) &
            (df['T_days'] <= MAX_DAYS_TO_EXPIRY)]

    df = df.sort_values(['Date', 'T_days', 'Strike']).reset_index(drop=True)
    print(f"  After filters: {len(df):,} rows")
    return df

# ================================================================
# SECTION 4: ATM SELECTION
# ================================================================

def select_atm(df, label):
    """One ATM option per trading day — nearest strike to spot."""
    df = df.copy()
    df['ATM_dist'] = abs(df['Spot'] - df['Strike'])
    df = df.sort_values(['Date', 'T_days', 'ATM_dist'])
    atm = df.groupby('Date').first().reset_index()
    atm = atm.drop(columns=['ATM_dist'])
    print(f"  {label} ATM: {len(atm)} trading days")
    return atm

# ================================================================
# SECTION 5: ADD COMPUTED COLUMNS
# ================================================================

def add_columns(df):
    df['T_years']   = df['T_days'] / 252
    df['Moneyness'] = df['Spot'] / df['Strike']
    df['r']         = df['Date'].apply(get_risk_free_rate)
    df = df.rename(columns={'Close': 'C_market'})
    return df

# ================================================================
# SECTION 6: ADD REALIZED VOLATILITY
# ================================================================

def add_rv(df, spot_file):
    """22-day realized volatility from spot data (thesis eq 3.17)."""
    print(f"  Loading spot data: {spot_file}")
    sp = pd.read_csv(spot_file)

    # Handle timezone-aware dates from yfinance
    sp['Date'] = pd.to_datetime(sp['Date'], utc=True)
    sp['Date'] = sp['Date'].dt.tz_convert('Asia/Kolkata').dt.date
    sp['Date'] = pd.to_datetime(sp['Date'])

    sp = sp[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
    sp['Log_Return'] = np.log(sp['Close'] / sp['Close'].shift(1))
    sp['RV_22'] = np.sqrt(
        sp['Log_Return'].pow(2).rolling(22).mean() * 252)

    print(f"  Spot: {len(sp)} days | RV mean = {sp['RV_22'].mean()*100:.2f}%")

    df = df.merge(sp[['Date', 'RV_22']], on='Date', how='left')
    missing = df['RV_22'].isna().sum()
    if missing > 0:
        print(f"  Dropping {missing} rows: no RV match in spot data")
        df = df.dropna(subset=['RV_22'])
    return df

# ================================================================
# SECTION 7: DIAGNOSTICS
# ================================================================

def diagnostics(df, label):
    print(f"\n  ── {label} ──────────────────────────────────")
    print(f"  Rows          : {len(df)}")
    print(f"  Date range    : {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"  Spot range    : {df['Spot'].min():.0f} to {df['Spot'].max():.0f}")
    print(f"  Mean moneyness: {df['Moneyness'].mean():.4f}  (expect ~1.0)")
    print(f"  Mean T_days   : {df['T_days'].mean():.1f}")
    print(f"  Mean C_market : {df['C_market'].mean():.2f} Rs")
    print(f"  Mean RV_22    : {df['RV_22'].mean()*100:.2f}%")
    atm_pct = ((df['Moneyness'] >= 0.95) &
                (df['Moneyness'] <= 1.05)).mean() * 100
    print(f"  Within 5% ATM : {atm_pct:.1f}% of rows")

# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 0: DATA CLEANING & ATM FILTERING")
    print("=" * 60)

    print("\n[1/5] Loading CALL options...")
    df_ce = load_options(CALL_FILES, 'CE')

    print("\n[2/5] Loading PUT options...")
    df_pe = load_options(PUT_FILES, 'PE')

    print("\n[3/5] Selecting ATM per day...")
    df_ce = select_atm(df_ce, 'CE')
    df_pe = select_atm(df_pe, 'PE')

    print("\n[4/5] Adding computed columns...")
    df_ce = add_columns(df_ce)
    df_pe = add_columns(df_pe)

    print("\n[5/5] Adding realized volatility...")
    df_ce = add_rv(df_ce, SPOT_FILE)
    df_pe = add_rv(df_pe, SPOT_FILE)

    diagnostics(df_ce, "ATM CALLS (CE)")
    diagnostics(df_pe, "ATM PUTS  (PE)")

    # Merge CE and PE — CE is primary, PE available for cross-check
    df_pe_sub = df_pe[['Date', 'C_market', 'Strike']].rename(
        columns={'C_market': 'C_market_PE', 'Strike': 'Strike_PE'})
    df_baseline = df_ce.merge(df_pe_sub, on='Date', how='inner')
    print(f"\n  Baseline (CE+PE merged): {len(df_baseline)} trading days")

    # Save
    df_ce.to_csv('nifty_atm_calls_clean.csv',        index=False)
    df_pe.to_csv('nifty_atm_puts_clean.csv',         index=False)
    df_baseline.to_csv('nifty_atm_options_baseline.csv', index=False)

    print("\n" + "=" * 60)
    print("STEP 0 COMPLETE — Files saved:")
    print("=" * 60)
    print(f"  nifty_atm_calls_clean.csv       {len(df_ce)} rows")
    print(f"  nifty_atm_puts_clean.csv        {len(df_pe)} rows")
    print(f"  nifty_atm_options_baseline.csv  {len(df_baseline)} rows")
    print("""
Columns in baseline:
  Date, Expiry, Option_Type, Strike, Spot, C_market,
  T_days, T_years, Moneyness, r, RV_22, C_market_PE, Strike_PE

Next: run step1_bs_baseline_realdata.py
""")
