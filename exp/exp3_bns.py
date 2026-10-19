"""
exp/exp3_bns.py
===============
Experiment 3: Calibration of the three-asset BNS generalized variance
formula to empirical determinant series.

Fits the BNS expected generalized realized variance (Theorem 4.1) to the
empirical 10-day rolling determinant-based generalized variance series for
each fixed 3-stock portfolio by nonlinear least squares.

Reduced parameterization (8 free parameters):
  [lambda, kappa11, kappa12, kappa13, sigma10_sq, sigma20_sq, sigma30_sq, kappa02]

Fixed: rho = [1,1,1]/sqrt(3), kappa_i2 = [0.1, 0.1, 0.1].

The 1/2 factor inconsistency in paper Eq. (4.5) is dropped for internal
consistency; see bns.py for the correction documentation.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import PORTFOLIOS
from src.bns import calibrate_bns, bns_expected_gen_variance, bns_inference
from src.metrics import compute_all_metrics
from src.diagnostics import plot_fitted_vs_realized

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp3(exp1_results: dict | None = None) -> dict:
    """Run experiment 3: BNS model calibration.

    Parameters
    ----------
    exp1_results:
        Output from :func:`exp1_construction.run_exp1`. If ``None``,
        the experiment reloads saved CSV artifacts from the results directory.

    Returns
    -------
    dict
        Portfolio-keyed dict of calibration results and error metrics.
    """
    logger.info("=== Experiment 3: BNS Calibration ===")

    if exp1_results is None:
        logger.info("Loading Exp-1 artifacts from %s …", RESULTS_DIR)
        exp1_results = _load_exp1_artifacts()

    all_results: dict = {}

    for group_name, res in exp1_results.items():
        tickers = PORTFOLIOS[group_name]
        logger.info("--- Portfolio: %s (%s) ---", group_name, tickers)

        det_series: pd.Series = res["det_rolling"]
        corr_matrix: np.ndarray = res["corr_matrix"]
        T_array: np.ndarray = res["T_rolling"]
        returns: pd.DataFrame = res["returns"]

        # First-window (first 10 daily returns) per-asset sample variances:
        # used as warm-start sigma0_sq so sigma0 ≠ kappa1 from iteration 0,
        # preventing the degenerate flat-curve local minimum.
        sigma0_sq_init: np.ndarray = returns.iloc[:10].var(ddof=1).values

        logger.info(
            "  Target series: %d obs, corr_det = %.4e, sigma0_sq_init = %s",
            len(det_series), np.linalg.det(corr_matrix), sigma0_sq_init,
        )

        # -------------------------------------------------------------------
        # Calibrate BNS by NLS (multi-start)
        # -------------------------------------------------------------------
        logger.info("  Running BNS NLS calibration …")
        calib = calibrate_bns(
            det_series=det_series,
            T_array=T_array,
            corr_matrix=corr_matrix,
            n_starts=3,
            seed=42,
            sigma0_sq_init=sigma0_sq_init,
        )

        # -------------------------------------------------------------------
        # Inference
        # -------------------------------------------------------------------
        infer = bns_inference(calib)

        # -------------------------------------------------------------------
        # Error metrics
        # -------------------------------------------------------------------
        y_obs = np.asarray(det_series, dtype=np.float64)
        fitted = calib["fitted_values"]
        metrics = compute_all_metrics(y_obs, fitted, label=f"BNS-{group_name}")

        # -------------------------------------------------------------------
        # Log results
        # -------------------------------------------------------------------
        logger.info("  BNS RMSE = %.6e", metrics["rmse"])
        logger.info("  BNS APE  = %.6f", metrics["ape"])
        logger.info("  BNS AAE  = %.6e", metrics["aae"])
        logger.info("  BNS R²   = %.6f", metrics["r_squared"])
        logger.info("  Parameters:")
        for name, val, se, tv, pv in zip(
            infer["param_names"],
            infer["params"],
            infer["std_errors"],
            infer["t_values"],
            infer["p_values"],
        ):
            logger.info(
                "    %s = %.6e  SE=%.4e  t=%.3f  p=%.4f", name, val, se, tv, pv
            )

        # -------------------------------------------------------------------
        # Save fitted-vs-realized plot
        # -------------------------------------------------------------------
        plot_fitted_vs_realized(
            det_series=det_series,
            fitted_heston=np.full_like(fitted, np.nan),  # Heston added in exp4
            fitted_bns=fitted,
            group_name=group_name,
            save_path=os.path.join(
                RESULTS_DIR, f"exp3_bns_fit_{group_name}.png"
            ),
        )

        # -------------------------------------------------------------------
        # Save CSV of fitted values
        # -------------------------------------------------------------------
        df_out = pd.DataFrame(
            {"realized": y_obs, "bns_fitted": fitted, "residual": y_obs - fitted},
            index=det_series.index,
        )
        df_out.to_csv(
            os.path.join(RESULTS_DIR, f"exp3_bns_fitted_{group_name}.csv")
        )

        all_results[group_name] = {
            "calib": calib,
            "infer": infer,
            "metrics": metrics,
            "fitted_values": fitted,
            "det_series": det_series,
            "T_array": T_array,
            "corr_matrix": corr_matrix,
        }

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    _print_summary_table(all_results)
    _save_summary_csv(all_results, tag="bns")

    logger.info("=== Experiment 3 complete ===")
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_exp1_artifacts() -> dict:
    """Reload experiment-1 data from loader/processor (same as exp2 helper)."""
    from data.loader import END_DATE, START_DATE, load_all_portfolios
    from data.processor import compute_full_sample_correlation, compute_log_returns
    from src.covariance import compute_horizon_map, rolling_covariance_determinants

    all_prices = load_all_portfolios(start=START_DATE, end=END_DATE)
    out: dict = {}
    for group_name, prices in all_prices.items():
        returns = compute_log_returns(prices)
        corr_matrix = compute_full_sample_correlation(returns)
        det_rolling = rolling_covariance_determinants(returns, window=10)
        T_rolling = compute_horizon_map(det_rolling)
        out[group_name] = {
            "prices": prices,
            "returns": returns,
            "corr_matrix": corr_matrix,
            "det_rolling": det_rolling,
            "det_block": det_rolling,
            "T_rolling": T_rolling,
            "T_block": T_rolling,
        }
    return out


def _print_summary_table(all_results: dict) -> None:
    rows = []
    for group_name, res in all_results.items():
        m = res["metrics"]
        rows.append(
            {
                "portfolio": group_name,
                "tickers": str(PORTFOLIOS[group_name]),
                "rmse": f"{m['rmse']:.6e}",
                "ape": f"{m['ape']:.4f}",
                "aae": f"{m['aae']:.6e}",
                "r_squared": f"{m['r_squared']:.4f}",
            }
        )
    df = pd.DataFrame(rows).set_index("portfolio")
    logger.info("\n=== BNS Calibration Summary ===\n%s", df.to_string())


def _save_summary_csv(all_results: dict, tag: str) -> None:
    rows = []
    for group_name, res in all_results.items():
        m = res["metrics"]
        infer = res["infer"]
        row = {"portfolio": group_name, "model": tag}
        row.update({k: v for k, v in m.items() if k != "label"})
        for name, val, pv in zip(
            infer["param_names"], infer["params"], infer["p_values"]
        ):
            row[name] = val
            row[f"{name}_pval"] = pv
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f"exp3_{tag}_summary.csv"), index=False
    )


if __name__ == "__main__":
    run_exp3()
