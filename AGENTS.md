# Project: Determinant-Based Generalized Realized Variance (Heston)

## Structure
```
data/
  loader.py      – Yahoo Finance downloader (PORTFOLIOS constant, download_prices, load_portfolio_prices, load_all_portfolios)
  processor.py   – log-returns, correlation/covariance, PSD validation, portfolio_metadata
src/
  covariance.py  – rolling and non-overlapping covariance determinant series, diagnostics, horizon map
  heston.py      – Heston GRV formula (8-term analytic expansion), NLS calibration, inference
exp/             – experiment scripts (user-created)
results/         – output artefacts (user-created)
tests/           – test files (user-created)
```

## Key constants (data/loader.py)
- `PORTFOLIOS` = {group1: [KO, AAPL, TSLA], group2: [GOOGL, MSFT, META], group3: [JPM, NVDA, AMZN]}
- `START_DATE = '2020-01-01'`, `END_DATE = '2024-01-01'`

## yfinance API notes (v1.6.0)
- `yf.download()` always returns MultiIndex `(Price, Ticker)` columns for both single and multiple tickers.
- With `auto_adjust=False`: includes `'Adj Close'` and `'Close'` (unadjusted); default (`auto_adjust=True`) returns only `'Close'` (adjusted).
- `data['Close']` → DataFrame with tickers as columns.

## Heston model conventions
- Reduced model used in calibration: `sigma0_sq = theta` (initial variance = long-run mean), so all transient `b_i` terms vanish and the GRV is constant in T: `E[σ_R²](T) = |C| · θ₁θ₂θ₃`.
- Full model (general `sigma0_sq`): 8-term analytic expansion of `∏(θᵢ + bᵢ·e^{-κᵢt})`.
- `calibrate_heston` bounds: κᵢ ∈ (1e-4, 50), θᵢ ∈ (1e-12, 1.0).
- `heston_inference` falls back to `np.linalg.pinv` when J^TJ is singular (common in reduced model since kappa is unidentifiable).

## Covariance estimators
- All use `ddof=1` (unbiased).
- `rolling_covariance_determinants`: length = `n - window + 1`, indexed by end-of-window date.
- `nonoverlapping_covariance_determinants`: length = `n // window`, incomplete tail discarded.
- `compute_horizon_map`: T_j = j/252 (annual, 252 trading-day convention), 1-indexed.

## Dependencies
- numpy, pandas, scipy, yfinance (v1.6.0)

---

## BNS model (src/bns.py)
- **Moments**: `bns_mean_variance` → `m_i(t) = exp(-lam*t)*(sigma_i0_sq-kappa_i1)+kappa_i1`;
  `bns_var_variance` → `(kappa_i2/2)*(1-exp(-2*lam*t))`;
  `bns_expected_sigma` → Brockhaus-Long: `sqrt(m) - kappa_i2*(1-exp(-2*lam*t))/(16*m^1.5)`,
  m clipped to 1e-20.  **BL valid only when** `kappa_i2 << 2*kappa_i1^2`; the fixed
  `kappa_i2=0.1` can violate this for small means, causing negative fitted variance.
- **Analytic integral** `bns_product_mean_integral`: 2^3=8-term binary subset expansion of
  `∏(a_i + b_i*exp(-lam*t))`.  Each term integrates to `coeff*(1-exp(-n*lam*T))/(n*lam)`.
  Verified vs numerical quad to < 1e-10 relative error.
- **Off-diagonal integral** `bns_off_diagonal_integral`: `scipy.integrate.quad` with
  `limit=50, epsabs=1e-10, epsrel=1e-8`.  Integrand uses **inline scalar math**
  (pre-extracted floats, no `np.array([t])`) to minimise per-call overhead.
  ~0.05 ms per call; calibration with near-true start needs ~18 residual evals (<0.1 s).
  Random starts from wide prior can need thousands of evals (O(minutes)); seed/scale matters.
- **Formula**: `E[σ_R²](T) = |R^B|/T * (E0 + lam*kappa02*total_correction)`.
  Diagonal correction = `sum_i δ_{ii}*ρ_i^2 * E0`.
  Off-diagonal = `sum_{i<j} 2*δ_{ij}*ρ_i*ρ_j * ∫E[σ_i]*E[σ_j]*m_k dt`.
- **Calibration**: 8 free params `[lam, κ_11, κ_12, κ_13, σ_10², σ_20², σ_30², κ_02]`;
  fixed `rho=[1,1,1]/√3`, `kappa_i2=[0.1,0.1,0.1]`; bounds as above.
  Multi-start TRF NLS, `max_nfev=20000`, keeps lowest cost.
- **Inference**: `bns_inference` raises `ValueError` if n ≤ 8; uses `(J^TJ)^{-1}` (pinv fallback).

## Error metrics (src/metrics.py)
- `compute_ape` / `compute_arpe` are identical (ARPE delegates); both skip `y_true==0`.
- `compute_r_squared` returns nan if SS_tot==0 (constant series).
- `percent_improvement(heston, bns)`: positive = BNS better; nan if heston==0.

## Diagnostics (src/diagnostics.py)
- `matplotlib.use('Agg')` at module top — must precede any `import matplotlib.pyplot`.
- All functions: `plt.tight_layout()` → `fig.savefig(dpi=150, bbox_inches='tight')` → `plt.close('all')`.
- `squeeze=False` everywhere → axes always 2D `(nrows, ncols)` array.
- `plot_det_series`: auto-selects log y-scale when `max_positive/min_positive > 100`.
- `plot_model_comparison`: detects group column (first non-metric/non-model col) and
  `'model'` column for hue; raises `ValueError` if no recognised metric column present.
