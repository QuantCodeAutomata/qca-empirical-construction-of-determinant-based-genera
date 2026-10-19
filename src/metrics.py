"""Error metric functions for evaluating generalized realized variance model fits.

Provides scalar summary statistics comparing observed and predicted series,
plus a convenience wrapper that computes all metrics at once and a utility
for computing the percentage improvement of one model over another.
"""

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.

    Returns
    -------
    float
        sqrt(mean((y_true - y_pred)^2)).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (MAPE), skipping zero true values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.

    Returns
    -------
    float
        mean(|y_true - y_pred| / |y_true|) over elements where y_true != 0.
        Returns nan if all true values are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0.0
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def compute_aae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (MAE / average absolute error).

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.

    Returns
    -------
    float
        mean(|y_true - y_pred|).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_arpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute relative percentage error (ARPE).

    Uses the same convention as APE: mean(|(y_true - y_pred) / y_true|),
    skipping observations where y_true = 0.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.

    Returns
    -------
    float
        mean(|(y_true - y_pred) / y_true|) over elements where y_true != 0.
    """
    return compute_ape(y_true, y_pred)


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.

    Returns
    -------
    float
        1 - SS_res / SS_tot, where SS_tot = sum((y_true - mean(y_true))^2).
        Returns nan if SS_tot = 0 (constant observed series).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = '',
) -> dict:
    """Compute RMSE, APE, AAE, ARPE, and R-squared in a single call.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed values.
    y_pred : np.ndarray
        Array of predicted/fitted values.
    label : str, optional
        Human-readable label identifying the metric set (e.g. model name or
        portfolio group).

    Returns
    -------
    dict
        Keys: ``rmse``, ``ape``, ``aae``, ``arpe``, ``r_squared``, ``label``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'rmse':      compute_rmse(y_true, y_pred),
        'ape':       compute_ape(y_true, y_pred),
        'aae':       compute_aae(y_true, y_pred),
        'arpe':      compute_arpe(y_true, y_pred),
        'r_squared': compute_r_squared(y_true, y_pred),
        'label':     label,
    }


def percent_improvement(metric_heston: float, metric_bns: float) -> float:
    """Percentage improvement of BNS over Heston for a given error metric.

    A positive value means BNS has a lower (better) error than Heston.

    Parameters
    ----------
    metric_heston : float
        Error metric value for the Heston model.
    metric_bns : float
        Error metric value for the BNS model.

    Returns
    -------
    float
        100 * (metric_heston - metric_bns) / metric_heston.
        Returns nan if metric_heston = 0.
    """
    if metric_heston == 0.0:
        return float('nan')
    return 100.0 * (metric_heston - metric_bns) / metric_heston
