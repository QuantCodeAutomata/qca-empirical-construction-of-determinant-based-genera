"""
data/processor.py
=================
Data-processing utilities for the determinant-based generalized realized
variance experiment.

All matrix computations use ``float64`` precision.  Functions validate their
outputs and raise descriptive errors on malformed inputs.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from a price DataFrame.

    The log-return on day *t* is defined as

    .. math::

        r_{i,t} = \\log\\bigl(S_{i,t} / S_{i,t-1}\\bigr)

    The first row (which would be all-NaN after the shift) is dropped.

    Parameters
    ----------
    prices:
        Price DataFrame with assets as columns and a ``DatetimeIndex``.
        Must contain at least two rows and no negative prices.

    Returns
    -------
    pd.DataFrame
        Log-return DataFrame of shape ``(len(prices) - 1, n_assets)`` with
        the same column labels as *prices*.

    Raises
    ------
    ValueError
        If *prices* has fewer than two rows or contains non-positive values.
    """
    if len(prices) < 2:
        raise ValueError(
            f"prices must have at least 2 rows; got {len(prices)}."
        )
    if (prices.values <= 0).any():
        raise ValueError(
            "prices contains non-positive values; log-returns are undefined."
        )

    log_returns: pd.DataFrame = np.log(prices / prices.shift(1))
    # Drop the first row (all NaN from the shift).
    log_returns = log_returns.iloc[1:].astype(np.float64)

    logger.debug(
        "compute_log_returns: prices %s → returns %s",
        prices.shape,
        log_returns.shape,
    )
    return log_returns


# ---------------------------------------------------------------------------
# Full-sample second-moment estimation
# ---------------------------------------------------------------------------


def validate_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether *matrix* is positive semidefinite.

    Uses ``numpy.linalg.eigvalsh``, which is numerically stable for symmetric
    matrices.

    Parameters
    ----------
    matrix:
        Square, real-valued array.
    tol:
        Tolerance; a matrix is considered PSD when all eigenvalues
        are ≥ ``-tol``.  Defaults to ``1e-8``.

    Returns
    -------
    bool
        ``True`` if every eigenvalue of *matrix* is ≥ ``-tol``.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    return bool(np.all(eigenvalues >= -tol))


def compute_full_sample_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Compute the full-sample Pearson correlation matrix.

    Parameters
    ----------
    returns:
        Log-return DataFrame produced by :func:`compute_log_returns`.
        Must have at least two columns and two rows.

    Returns
    -------
    np.ndarray
        ``(n_assets, n_assets)`` correlation matrix in ``float64``.

    Raises
    ------
    ValueError
        If the resulting matrix is not symmetric, does not have ones on the
        diagonal, or is not positive semidefinite (within tolerance ``1e-8``).
    """
    if returns.shape[0] < 2:
        raise ValueError(
            f"returns must have at least 2 rows to compute correlation; "
            f"got {returns.shape[0]}."
        )

    corr: np.ndarray = returns.corr().values.astype(np.float64)

    if not np.allclose(corr, corr.T, atol=1e-10):
        raise ValueError("Computed correlation matrix is not symmetric.")

    if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
        raise ValueError("Computed correlation matrix does not have ones on the diagonal.")

    if not validate_psd(corr, tol=1e-8):
        raise ValueError("Computed correlation matrix is not positive semidefinite.")

    logger.debug("Full-sample correlation matrix (shape %s) validated.", corr.shape)
    return corr


def compute_full_sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute the full-sample sample covariance matrix.

    The estimator uses the unbiased (``ddof=1``) convention, which is
    ``pandas.DataFrame.cov``'s default.

    Parameters
    ----------
    returns:
        Log-return DataFrame produced by :func:`compute_log_returns`.

    Returns
    -------
    np.ndarray
        ``(n_assets, n_assets)`` covariance matrix in ``float64``.

    Raises
    ------
    ValueError
        If *returns* has fewer than two rows.
    """
    if returns.shape[0] < 2:
        raise ValueError(
            f"returns must have at least 2 rows to compute covariance; "
            f"got {returns.shape[0]}."
        )

    cov: np.ndarray = returns.cov().values.astype(np.float64)
    logger.debug("Full-sample covariance matrix (shape %s) computed.", cov.shape)
    return cov


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def portfolio_metadata(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    group_name: str,
) -> dict:
    """Collect summary metadata for a portfolio.

    Parameters
    ----------
    prices:
        Raw price DataFrame (before log-return computation).
    returns:
        Log-return DataFrame produced by :func:`compute_log_returns`.
    group_name:
        Human-readable label for the portfolio group.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``group_name`` – portfolio label.
        * ``n_observations`` – number of return observations.
        * ``tickers`` – list of asset ticker symbols.
        * ``start_date`` – first date in the price index.
        * ``end_date`` – last date in the price index.
        * ``any_missing_prices`` – ``True`` if *prices* contains any NaN.
    """
    return {
        "group_name": group_name,
        "n_observations": len(returns),
        "tickers": list(prices.columns),
        "start_date": prices.index[0],
        "end_date": prices.index[-1],
        "any_missing_prices": bool(prices.isna().any().any()),
    }
