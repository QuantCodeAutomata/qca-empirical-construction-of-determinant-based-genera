"""
exp/run_all.py
==============
Master script that runs all four experiments in sequence, passing outputs
from earlier experiments directly to later ones to avoid redundant data
downloads.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute all experiments sequentially."""
    logger.info("========================================")
    logger.info("  Determinant-Based Generalized Realized")
    logger.info("  Variance — Full Experiment Pipeline")
    logger.info("========================================")

    from exp.exp1_construction import run_exp1
    from exp.exp2_heston import run_exp2
    from exp.exp3_bns import run_exp3
    from exp.exp4_comparison import run_exp4

    # Exp 1: Build data
    logger.info("\n[1/4] Running Experiment 1: Data Construction …")
    exp1_results = run_exp1()

    # Exp 2: Heston calibration
    logger.info("\n[2/4] Running Experiment 2: Heston Calibration …")
    heston_results = run_exp2(exp1_results=exp1_results)

    # Exp 3: BNS calibration
    logger.info("\n[3/4] Running Experiment 3: BNS Calibration …")
    bns_results = run_exp3(exp1_results=exp1_results)

    # Exp 4: Comparison + robustness
    logger.info("\n[4/4] Running Experiment 4: Model Comparison …")
    comparison_results = run_exp4(
        exp1_results=exp1_results,
        heston_results=heston_results,
        bns_results=bns_results,
    )

    logger.info("\n========================================")
    logger.info("  All experiments complete.")
    logger.info("  Results saved in: results/")
    logger.info("========================================")


if __name__ == "__main__":
    main()
