"""
exp/exp1_construction.py
========================
Experiment 1: Empirical construction of determinant-based generalized
realized variance for three fixed 3-stock portfolios.

Reproduces the paper's empirical generalized variance measure by:
  - Downloading daily Close prices for 9 assets (2020-01-01 to 2024-01-01)
  - Computing daily log-returns for each portfolio
  - Estimating 3x3 covariance matrices over 10-day intervals
  - Taking covariance determinants as the realized generalized variance proxy
  - Producing descriptive diagnostics (return distributions, heatmaps,
    cumulative returns, determinant series)

Rolling 10-day windows are the primary specification; non-overlapping blocks
are computed as a robustness variant.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import END_DATE, PORTFOLIOS, START_DATE, load_all_portfolios
from data.processor import (
    compute_full_sample_correlation,
    compute_full_sample_covariance,
    compute_log_returns,
    portfolio_metadata,
    validate_psd,
)
from src.covariance import (
    covariance_matrix_diagnostics,
    compute_horizon_map,
    nonoverlapping_covariance_determinants,
    rolling_covariance_determinants,
)
from src.diagnostics import (
    plot_correlation_heatmaps,
    plot_cumulative_returns,
    plot_det_series,
    plot_return_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW = 10  # trading days per covariance estimation window


def run_exp1() -> dict:
    """Run experiment 1: construct empirical determinant-based generalized variance.

    Returns
    -------
    dict
        Portfolio-keyed dict containing returns, correlation matrices,
        rolling/block determinant series, horizon maps, and metadata.
    """
    logger.info("=== Experiment 1: Data Construction ===")
    logger.info("Portfolios: %s", PORTFOLIOS)
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("Window size: %d trading days", WINDOW)

    # -----------------------------------------------------------------------
    # Step 1: Download daily Close prices
    # -----------------------------------------------------------------------
    logger.info("Step 1: Downloading prices …")
    all_prices = load_all_portfolios(start=START_DATE, end=END_DATE, price_col="Close")

    results: dict = {}
    returns_dict: dict = {}
    corr_dict: dict = {}
    det_rolling_dict: dict = {}
    det_block_dict: dict = {}

    for group_name, prices in all_prices.items():
        logger.info("--- Portfolio: %s (%s) ---", group_name, PORTFOLIOS[group_name])

        # -------------------------------------------------------------------
        # Step 2 & 3: Align series on common trading dates, compute log-returns
        # -------------------------------------------------------------------
        returns = compute_log_returns(prices)
        logger.info("  Returns shape: %s (after dropping first NaN row)", returns.shape)

        # -------------------------------------------------------------------
        # Step 4: Full-sample correlation matrix
        # -------------------------------------------------------------------
        corr_matrix = compute_full_sample_correlation(returns)
        cov_matrix_full = compute_full_sample_covariance(returns)
        corr_det = float(np.linalg.det(corr_matrix))
        logger.info("  Full-sample |C| = %.6e", corr_det)
        logger.info("  PSD check: %s", validate_psd(corr_matrix))

        # -------------------------------------------------------------------
        # Step 5: Rolling 10-day covariance determinants (primary)
        # -------------------------------------------------------------------
        det_rolling = rolling_covariance_determinants(returns, window=WINDOW)
        T_rolling = compute_horizon_map(det_rolling)
        logger.info(
            "  Rolling det series: %d observations, range [%.3e, %.3e]",
            len(det_rolling), det_rolling.min(), det_rolling.max(),
        )

        # -------------------------------------------------------------------
        # Step 6: Non-overlapping 10-day blocks (sensitivity variant)
        # -------------------------------------------------------------------
        det_block = nonoverlapping_covariance_determinants(returns, window=WINDOW)
        T_block = compute_horizon_map(det_block)
        logger.info(
            "  Block det series: %d observations, range [%.3e, %.3e]",
            len(det_block), det_block.min(), det_block.max(),
        )

        # -------------------------------------------------------------------
        # Step 7: Diagnostics DataFrame
        # -------------------------------------------------------------------
        diag_rolling = covariance_matrix_diagnostics(returns, window=WINDOW, variant="rolling")
        diag_block = covariance_matrix_diagnostics(returns, window=WINDOW, variant="nonoverlapping")
        n_singular_rolling = int(diag_rolling["is_singular"].sum())
        n_singular_block = int(diag_block["is_singular"].sum())
        logger.info(
            "  Singular matrices — rolling: %d, block: %d",
            n_singular_rolling, n_singular_block,
        )

        # -------------------------------------------------------------------
        # Portfolio metadata
        # -------------------------------------------------------------------
        meta = portfolio_metadata(prices, returns, group_name)
        meta.update(
            {
                "corr_det": corr_det,
                "window": WINDOW,
                "n_rolling_det": len(det_rolling),
                "n_block_det": len(det_block),
                "n_singular_rolling": n_singular_rolling,
                "n_singular_block": n_singular_block,
                "corr_matrix": corr_matrix.tolist(),
                "cov_matrix_full": cov_matrix_full.tolist(),
            }
        )

        results[group_name] = {
            "prices": prices,
            "returns": returns,
            "corr_matrix": corr_matrix,
            "cov_matrix_full": cov_matrix_full,
            "corr_det": corr_det,
            "det_rolling": det_rolling,
            "det_block": det_block,
            "T_rolling": T_rolling,
            "T_block": T_block,
            "diag_rolling": diag_rolling,
            "diag_block": diag_block,
            "metadata": meta,
        }

        returns_dict[group_name] = returns
        corr_dict[group_name] = corr_matrix
        det_rolling_dict[group_name] = det_rolling
        det_block_dict[group_name] = det_block

    # -----------------------------------------------------------------------
    # Step 8: Produce diagnostics and save outputs
    # -----------------------------------------------------------------------
    logger.info("Step 8: Generating diagnostic plots …")

    plot_return_distributions(
        returns_dict,
        save_path=os.path.join(RESULTS_DIR, "exp1_return_distributions.png"),
    )
    plot_correlation_heatmaps(
        corr_dict,
        save_path=os.path.join(RESULTS_DIR, "exp1_correlation_heatmaps.png"),
    )
    plot_cumulative_returns(
        returns_dict,
        save_path=os.path.join(RESULTS_DIR, "exp1_cumulative_returns.png"),
    )
    plot_det_series(
        {"rolling_" + k: v for k, v in det_rolling_dict.items()},
        save_path=os.path.join(RESULTS_DIR, "exp1_det_series_rolling.png"),
    )
    plot_det_series(
        {"block_" + k: v for k, v in det_block_dict.items()},
        save_path=os.path.join(RESULTS_DIR, "exp1_det_series_block.png"),
    )

    # -----------------------------------------------------------------------
    # Step 9: Save determinant series and metadata to CSV/JSON
    # -----------------------------------------------------------------------
    logger.info("Step 9: Saving data artifacts …")
    for group_name, res in results.items():
        tickers_str = "_".join(PORTFOLIOS[group_name])

        # Rolling determinant series
        res["det_rolling"].to_csv(
            os.path.join(RESULTS_DIR, f"exp1_{group_name}_det_rolling.csv"),
            header=True,
        )
        # Non-overlapping block determinant series
        res["det_block"].to_csv(
            os.path.join(RESULTS_DIR, f"exp1_{group_name}_det_block.csv"),
            header=True,
        )
        # Log-returns
        res["returns"].to_csv(
            os.path.join(RESULTS_DIR, f"exp1_{group_name}_returns.csv"),
        )

        # Metadata JSON (exclude non-serializable numpy arrays already converted)
        meta_out = {k: v for k, v in res["metadata"].items()}
        with open(
            os.path.join(RESULTS_DIR, f"exp1_{group_name}_metadata.json"), "w"
        ) as fh:
            json.dump(meta_out, fh, indent=2, default=str)

        logger.info(
            "  %s: saved returns (%d rows), rolling dets (%d), block dets (%d)",
            group_name,
            len(res["returns"]),
            len(res["det_rolling"]),
            len(res["det_block"]),
        )

    logger.info("=== Experiment 1 complete ===")
    return results


if __name__ == "__main__":
    run_exp1()
