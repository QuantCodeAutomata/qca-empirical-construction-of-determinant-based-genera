"""
src/covariance.py
=================
Covariance matrix estimation and determinant-series construction for the
determinant-based generalized realized variance experiment.

Two window variants are supported:

* **rolling** – each window advances one trading day at a time (overlapping).
* **nonoverlapping** – consecutive, non-overlapping blocks of ``window`` rows.

All sample covariance matrices use ``ddof=1`` (unbiased estimator).
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Accepted variant labels
_VARIANTS = frozenset({"rolling", "nonoverlapping"})


def _check_variant(variant: str) -> None:
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Supported variants: {sorted(_VARIANTS)}."
        )


def _check_returns(returns: pd.DataFrame, window: int) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pandas DataFrame; got {type(returns)}.")
    if returns.shape[1] < 1:
        raise ValueError("returns must have at least one column.")
    if window < 2:
        raise ValueError(f"window must be ≥ 2; got {window}.")
    if returns.shape[0] < window:
        raise ValueError(
            f"returns has {returns.shape[0]} rows but window={window}; "
            f"need at least {window} rows."
        )


# ---------------------------------------------------------------------------
# Determinant series
# ---------------------------------------------------------------------------


def rolling_covariance_determinants(
    returns: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """Compute rolling-window sample covariance determinants.

    For each date at index position *t* (starting from ``window - 1``),
    the sample covariance matrix is estimated from the ``window`` rows
    ``returns[t-window+1 : t+1]`` using ``ddof=1``.  The determinant is
    assigned to the end-of-window date ``returns.index[t]``.

    Parameters
    ----------
    returns:
        Log-return DataFrame, assets as columns.
    window:
        Rolling-window length in trading days; must be ≥ 2.

    Returns
    -------
    pd.Series
        Determinant values indexed by the end-of-window date,
        with ``name='det_cov'`` and ``dtype=float64``.
        Length = ``len(returns) - window + 1``.
    """
    _check_returns(returns, window)
    n = len(returns)
    dates: list = []
    dets: list[float] = []

    for t in range(window - 1, n):
        block: np.ndarray = returns.iloc[t - window + 1 : t + 1].values.astype(
            np.float64
        )
        cov: np.ndarray = np.cov(block.T, ddof=1)
        dets.append(float(np.linalg.det(cov)))
        dates.append(returns.index[t])

    series = pd.Series(dets, index=dates, name="det_cov", dtype=np.float64)
    logger.debug(
        "rolling_covariance_determinants: window=%d → %d observations",
        window,
        len(series),
    )
    return series


def nonoverlapping_covariance_determinants(
    returns: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """Compute non-overlapping block sample covariance determinants.

    Returns are split into consecutive, non-overlapping blocks of ``window``
    rows.  Incomplete trailing blocks are discarded.  The determinant of the
    block covariance matrix (``ddof=1``) is assigned to the last date of the
    block.

    Parameters
    ----------
    returns:
        Log-return DataFrame, assets as columns.
    window:
        Block length in trading days; must be ≥ 2.

    Returns
    -------
    pd.Series
        Determinant values indexed by the last date of each block,
        with ``name='det_cov'`` and ``dtype=float64``.
        Length = ``len(returns) // window``.
    """
    _check_returns(returns, window)
    n = len(returns)
    n_blocks: int = n // window
    dates: list = []
    dets: list[float] = []

    for b in range(n_blocks):
        start_idx = b * window
        end_idx = start_idx + window
        block: np.ndarray = returns.iloc[start_idx:end_idx].values.astype(np.float64)
        cov: np.ndarray = np.cov(block.T, ddof=1)
        dets.append(float(np.linalg.det(cov)))
        dates.append(returns.index[end_idx - 1])

    series = pd.Series(dets, index=dates, name="det_cov", dtype=np.float64)
    logger.debug(
        "nonoverlapping_covariance_determinants: window=%d, "
        "%d complete blocks (of %d rows), %d remainder row(s) discarded",
        window,
        n_blocks,
        n,
        n - n_blocks * window,
    )
    return series


# ---------------------------------------------------------------------------
# Windowed covariance list
# ---------------------------------------------------------------------------


def estimate_windowed_covariances(
    returns: pd.DataFrame,
    window: int = 10,
    variant: str = "rolling",
) -> list[tuple]:
    """Return a list of ``(date, cov_matrix)`` tuples for every window position.

    This is a diagnostic helper; it exposes the full covariance matrix for
    each window rather than only the determinant.

    Parameters
    ----------
    returns:
        Log-return DataFrame, assets as columns.
    window:
        Window length in trading days; must be ≥ 2.
    variant:
        ``'rolling'`` or ``'nonoverlapping'``.

    Returns
    -------
    list[tuple[datetime, np.ndarray]]
        Each element is ``(date, cov_matrix)`` where *date* is the
        end-of-window date and *cov_matrix* is a ``(n_assets, n_assets)``
        ``float64`` array.

    Raises
    ------
    ValueError
        If *variant* is not ``'rolling'`` or ``'nonoverlapping'``.
    """
    _check_variant(variant)
    _check_returns(returns, window)
    n = len(returns)
    result: list[tuple] = []

    if variant == "rolling":
        for t in range(window - 1, n):
            block = returns.iloc[t - window + 1 : t + 1].values.astype(np.float64)
            cov = np.cov(block.T, ddof=1)
            result.append((returns.index[t], cov))
    else:  # nonoverlapping
        n_blocks = n // window
        for b in range(n_blocks):
            start_idx = b * window
            end_idx = start_idx + window
            block = returns.iloc[start_idx:end_idx].values.astype(np.float64)
            cov = np.cov(block.T, ddof=1)
            result.append((returns.index[end_idx - 1], cov))

    return result


# ---------------------------------------------------------------------------
# Diagnostics DataFrame
# ---------------------------------------------------------------------------


def covariance_matrix_diagnostics(
    returns: pd.DataFrame,
    window: int = 10,
    variant: str = "rolling",
) -> pd.DataFrame:
    """Compute per-window diagnostic statistics for the covariance matrices.

    For each window position the following quantities are recorded:

    * ``det`` – determinant of the sample covariance matrix.
    * ``min_eigenvalue`` – smallest eigenvalue (``eigvalsh``).
    * ``condition_number`` – ratio of largest to smallest eigenvalue;
      ``inf`` when ``min_eigenvalue ≤ 0``.
    * ``is_singular`` – ``True`` when ``|det| < 1e-20``.
    * ``log_det`` – ``log(det)`` when ``det > 0``, otherwise ``NaN``.

    Parameters
    ----------
    returns:
        Log-return DataFrame, assets as columns.
    window:
        Window length in trading days; must be ≥ 2.
    variant:
        ``'rolling'`` or ``'nonoverlapping'``.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the end-of-window date.
    """
    _check_variant(variant)
    windowed = estimate_windowed_covariances(returns, window, variant)
    records: list[dict] = []

    for date, cov in windowed:
        det = float(np.linalg.det(cov))
        eigenvalues: np.ndarray = np.linalg.eigvalsh(cov)
        min_eig = float(eigenvalues.min())
        max_eig = float(eigenvalues.max())

        if min_eig > 0.0:
            condition_number = max_eig / min_eig
        else:
            condition_number = np.inf

        records.append(
            {
                "date": date,
                "det": det,
                "min_eigenvalue": min_eig,
                "condition_number": condition_number,
                "is_singular": abs(det) < 1e-20,
                "log_det": float(np.log(det)) if det > 0.0 else np.nan,
            }
        )

    df = pd.DataFrame(records).set_index("date")
    logger.debug(
        "covariance_matrix_diagnostics: variant=%s, window=%d → %d rows",
        variant,
        window,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Horizon (T_j) mapping
# ---------------------------------------------------------------------------


def compute_horizon_map(
    det_series: pd.Series,
    freq: str = "annual",
) -> np.ndarray:
    """Map each determinant observation to a horizon ``T_j = j / 252``.

    The horizon is measured in **years** on the assumption that there are 252
    trading days per year.  The first observation is assigned
    ``T_1 = 1/252``, the second ``T_2 = 2/252``, and so on.

    Parameters
    ----------
    det_series:
        Determinant series with a date index (e.g., from
        :func:`rolling_covariance_determinants`).
    freq:
        Frequency convention; currently only ``'annual'`` (252 trading days
        per year) is supported.

    Returns
    -------
    np.ndarray
        Array of horizon values with the same length as *det_series*,
        in ``float64``.

    Raises
    ------
    ValueError
        If *freq* is not ``'annual'``.
    """
    if freq != "annual":
        raise ValueError(
            f"Unknown freq '{freq}'. Currently only 'annual' is supported."
        )
    n = len(det_series)
    T_array: np.ndarray = np.arange(1, n + 1, dtype=np.float64) / 252.0
    return T_array
