# Determinant-Based Generalized Realized Variance

Empirical construction and calibration of determinant-based generalized
realized variance for three fixed 3-stock portfolios, comparing the
Heston and Barndorff-Nielsen–Shephard (BNS) stochastic-volatility models.

## Background

This repository replicates the empirical exercise from a paper on
generalized dispersion-swap pricing. The paper defines **generalized
realized variance** as the determinant of a multivariate return
covariance matrix estimated over rolling or non-overlapping 10-trading-day
windows. Two closed-form expected-generalized-variance models are then
calibrated to these empirical series:

- **Heston**: CIR variance dynamics with closed-form 8-term integral
  (paper Eq. 3.6).
- **BNS**: Lévy-subordinator variance dynamics with a jump-inclusive
  determinant correction (paper Theorem 4.1 / Eq. 4.5).

The paper's headline claim — BNS fits the determinant series better than
Heston — is independently evaluated across three portfolios and two
window-construction variants.

## Portfolio Groups

| Group | Tickers         |
|-------|-----------------|
| 1     | KO, AAPL, TSLA  |
| 2     | GOOGL, MSFT, META |
| 3     | JPM, NVDA, AMZN |

Data: Yahoo Finance daily Close prices, 2020-01-01 to 2024-01-01.

## Repository Structure

```
repo/
├── data/
│   ├── loader.py        # yfinance downloader + alignment
│   └── processor.py     # log-returns, covariance matrices, determinants
├── src/
│   ├── covariance.py    # rolling / block covariance + PSD diagnostics
│   ├── heston.py        # Heston GRV formula + NLS calibration
│   ├── bns.py           # BNS GRV formula + NLS calibration
│   ├── diagnostics.py   # all matplotlib/seaborn plots
│   └── metrics.py       # RMSE, APE, AAE, ARPE, % improvement
├── exp/
│   ├── exp1_construction.py   # Build empirical GRV target series
│   ├── exp2_heston.py         # Heston calibration
│   ├── exp3_bns.py            # BNS calibration
│   ├── exp4_comparison.py     # Model comparison + robustness
│   └── run_all.py             # Master orchestration script
├── tests/
│   ├── test_covariance.py
│   ├── test_heston.py
│   ├── test_bns.py
│   └── test_metrics.py
├── results/
│   └── RESULTS.md       # Summary of all experiment findings
├── README.md
├── .gitignore
└── requirements.txt
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Experiments

```bash
python -m exp.run_all
```

This runs all four experiments in sequence, saving plots and metrics to
`results/`.

### Run Individual Experiments

```bash
python -m exp.exp1_construction
python -m exp.exp2_heston
python -m exp.exp3_bns
python -m exp.exp4_comparison
```

### Run Tests

```bash
pytest tests/ -v
```

## Implementation Notes

### Data
- Yahoo Finance daily **Close** prices (not Adjusted Close) as the paper
  references series labeled `*.Close`.
- Rolling 10-trading-day windows are the **primary** specification.
- Non-overlapping 10-day blocks are computed as a robustness variant.
- All arithmetic is `float64`; determinants are preserved at full precision.

### Heston Model (Exp 2)
- Standard CIR variance process per asset.
- Covariance structure `Σ_t = D_t C D_t` (paper's correlation-factor form).
- Expected determinant expands into 8 exponential terms.
- Reduced parameterization: `σ²_{i,0} = θ_i` (6 free params per portfolio).
- NLS via `scipy.optimize.least_squares` with positivity bounds.

### BNS Model (Exp 3)
- Common mean-reversion rate `λ` across assets.
- Jump-inclusive covariance-rate matrix via matrix determinant lemma.
- Brockhaus–Long second-order approximation for `E[σ_{i,t}]`.
- Integrals `E₀–E₆` evaluated analytically (E₀–E₃) and via Gauss–Legendre
  quadrature (E₄–E₆).
- Note: the leading `1/2` factor in paper Eq. (4.5) is dropped here
  (internal inconsistency documented in `src/bns.py`).

### Known Paper Ambiguities
1. **Close vs Adjusted Close** — this replication uses raw Close.
2. **Rolling vs non-overlapping windows** — rolling is primary; block is
   robustness.
3. **Horizon mapping T_j** — `T_j = j * 10 / 252` for the j-th window.
4. **Initial variance σ²_{i,0}** — set equal to `θ_i` (paper omits treatment).
5. **BNS 1/2 factor** — dropped (inconsistent with appendix derivation).
6. **APE/AAE/ARPE definitions** — stated explicitly in `src/metrics.py`.

## Expected Results

| Portfolio      | Heston RMSE | BNS RMSE | BNS Improvement |
|----------------|-------------|----------|-----------------|
| KO/AAPL/TSLA   | ~0.25       | ~0.17    | ~33 %           |
| GOOGL/MSFT/META| ~0.32       | ~0.17    | ~47 %           |
| JPM/NVDA/AMZN  | ~5.18       | ~1.11    | ~79 %           |

See `results/RESULTS.md` for the actual computed values.
