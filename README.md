# NiftyPINN-IV

Physics-Informed Neural Network (PINN) framework for implied volatility extraction from NSE Nifty 50 index options, extending Panda (2015) by replacing classical Black-Scholes Newton-Raphson inversion with PDE-constrained deep learning.

## Project Identity

- Project: NiftyPINN-IV
- Repo: nifty-pinn-implied-volatility
- Domain: Quant research, derivatives pricing, PINNs, Indian equity markets
- Market: NSE Nifty 50 index options (near-month ATM baseline + cross-section surface extension)

## Why This Project

Classical implied volatility extraction solves one nonlinear equation per option quote:

- Find $\sigma$ such that $BS(\sigma)=C_{market}$

This project replaces that with an inverse PINN that jointly enforces:

1. Black-Scholes PDE consistency in the $(s,\tau)$ domain
2. Market price fit
3. Boundary conditions
4. Direct differentiable Black-Scholes pricing loss

This makes the extraction physics-aware and extensible from scalar IV to full volatility surface estimation.

## Academic Foundation

Base thesis:

- Siba Prasada Panda (2015), University of Hyderabad
- TH8745: Information Content and Predictability of Implied Volatility: Evidence from the Indian Equity Market

What the thesis established:

- Classical IV from Nifty options via Newton-Raphson BS inversion
- Information-content regression: $RV_{t+1}=\alpha+\beta IV_t+\epsilon$
- Forecasting comparison of IV vs MA, EWMA, GARCH, EGARCH
- Evidence that IV is informationally strong for future realized volatility

## Mathematical Setup

Black-Scholes pricing (thesis Eq. 3.12, 3.13):

$$
C = S N(d_1) - K e^{-rT} N(d_2), \quad
P = K e^{-rT} N(-d_2) - S N(-d_1)
$$

$$
d_1 = \frac{\ln(S/K) + (r+\sigma^2/2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
$$

Realized volatility (thesis Eq. 3.17):

$$
RV = \sqrt{252 \cdot \text{mean}(\log\_return^2)}
$$

Normalized PINN coordinates:

- $c=C/K$
- $s=S/K$
- $\tau=T-t$

Black-Scholes PDE in normalized form:

$$
-\frac{\partial c}{\partial \tau}
+ \frac{1}{2}\sigma^2 s^2\frac{\partial^2 c}{\partial s^2}
+ rs\frac{\partial c}{\partial s}
- rc = 0
$$

Training objective used in Step 2:

$$
\mathcal{L} =
\lambda_{pde}\mathcal{L}_{pde}
+ \lambda_{bc}\mathcal{L}_{bc}
+ \lambda_{data}\mathcal{L}_{data}
+ \lambda_{iv}\mathcal{L}_{iv}
+ \lambda_{prior}\mathcal{L}_{prior}
$$

## Repository Layout

Main scripts:

- `src/step0_data_cleaning.py`
- `src/step1_bs_baseline_realdata.py`
- `src/step2_pinn.py`
- `src/Step3_comparison.py`
- `src/Step4_information content.py`
- `src/Step5_predictability.py`
- `src/Step6_vol surface.py`

Utility scripts:

- `src/get_iv.py`
- `src/predict_iv.py`

Data and outputs:

- `data/`
- `src/results/csv and txt/`
- `src/results/images/`

## End-to-End Pipeline

1. **Step 0** - Clean raw option files, keep near-month contracts, select ATM, add $RV_{22}$
2. **Step 1** - Classical BS baseline (Newton-Raphson + Brent fallback)
3. **Step 2** - Inverse PINN IV extraction (single-window and rolling windows)
4. **Step 3** - Comparison: PINN IV vs Classical IV vs Realized Vol
5. **Step 4** - Information content regressions ($RV_{t+1}$ on IV$_t$)
6. **Step 5** - Predictability benchmarking vs MA/EWMA/GARCH/EGARCH
7. **Step 6** - Volatility surface extension $\sigma(s,\tau)$

## Setup

Create environment and install:

```bash
pip install numpy pandas scipy matplotlib torch statsmodels arch
```

You can also try:

```bash
pip install -r requirements.txt
```

## How To Run

From repo root:

```bash
cd src
python ../src/step0_data_cleaning.py
python ../src/step1_bs_baseline_realdata.py
python ../src/step2_pinn.py
python ../src/Step3_comparison.py
python "../src/Step4_information content.py"
python ../src/Step5_predictability.py
python "../src/Step6_vol surface.py"
```

Single-option IV calculator:

```bash
python ../src/get_iv.py --S 24500 --K 24500 --r 0.065 --T 14 --C 185.50
```

Prediction pipeline:

```bash
python ../src/predict_iv.py --mode both
```

### Step 3: Descriptive and Agreement

- Mean PINN IV: **9.7525%**
- Mean Classical IV: **10.0762%**
- Mean Realized Vol: **11.0642%**
- PINN vs Classical correlation: **0.9854**
- PINN vs Classical $R^2$: **0.9710**
- MAE(PINN, Classical): **0.3789 pp**

### Step 4: Information Content ($RV_{t+1}=\alpha+\beta IV_t+\epsilon$)

- PINN IV only: $\beta=1.0237$, $p=0.0462$, $R^2=0.2919$
- Classical IV only: $\beta=0.9544$, $p=0.0239$, $R^2=0.2982$
- Mincer-Zarnowitz test: fail to reject unbiasedness in this sample

### Step 5: Out-of-Sample Forecasting (70/30 split)

RMSE (lower is better):

- Classical IV: **1.9859%**
- PINN IV: **2.1673%**
- EWMA: **2.2177%**
- MA: **2.2405%**
- GARCH: **2.7707%**

Interpretation:

- PINN extraction closely tracks classical IV while remaining PDE-constrained.
- Both IV series carry significant information about future RV.
- In this short sample, Classical IV is slightly stronger out-of-sample; PINN remains competitive and extensible to surface modeling.

## Limitations

- Small effective sample in lead-lag regressions (12 observations after shifting)
- Relative-path script design (consider refactoring to config-based absolute/relative path handling)
- Surface tails can be noisy without stricter liquidity and no-arbitrage constraints

## Next Research Extensions

1. Add more years of Nifty options data to increase statistical power
2. Add no-arbitrage constraints on the learned surface (calendar + strike monotonicity/convexity)
3. Benchmark against SVI/SABR/local-vol models
4. Build live data ingestion and daily retraining workflow

## Citation

If you use this work, please cite:

- Panda, S. P. (2015). *Information Content and Predictability of Implied Volatility: Evidence from the Indian Equity Market*. University of Hyderabad, TH8745.
