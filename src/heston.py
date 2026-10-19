"""
src/heston.py
=============
Three-asset Heston stochastic-volatility model: expected generalized realized
variance (GRV) formula and non-linear least-squares (NLS) calibration.

Variance dynamics (CIR process for each asset *i*):

.. math::

    d\\sigma_i^2 = \\kappa_i (\\theta_i - \\sigma_i^2)\\,dt
                  + \\xi_i \\sigma_i\\,dW_i

Expected instantaneous variance:

.. math::

    E[\\sigma_{i,t}^2] = \\theta_i
                         + (\\sigma_{i,0}^2 - \\theta_i)\\,e^{-\\kappa_i t}

Covariance determinant:

.. math::

    |\\Sigma_t^H| = |C|\\,\\prod_{i=1}^{3} E[\\sigma_{i,t}^2]

Expected generalized realized variance (Eq. 3.6 in the paper):

.. math::

    E[\\sigma_R^2](T) = \\frac{|C|}{T}
        \\int_0^T \\prod_{i=1}^{3} E[\\sigma_{i,t}^2]\\,dt

The integrand is expanded analytically into 8 exponential terms; see
:func:`heston_expected_gen_variance`.

**Reduced model**: when calibrating via :func:`heston_residuals` /
:func:`calibrate_heston`, the initial variance is pinned to the long-run
mean (``sigma0_sq = theta``), which eliminates the transient terms and yields
a specification consistent with the paper's reduced-form interpretation.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-asset expected variance
# ---------------------------------------------------------------------------


def heston_expected_variance_i(
    t: np.ndarray,
    kappa_i: float,
    theta_i: float,
    sigma_i0_sq: float,
) -> np.ndarray:
    """Expected instantaneous variance for asset *i* at times *t*.

    .. math::

        E[\\sigma_{i,t}^2] = \\theta_i
                             + (\\sigma_{i,0}^2 - \\theta_i)\\,e^{-\\kappa_i t}

    Parameters
    ----------
    t:
        Array of time points (in years, ≥ 0).
    kappa_i:
        Mean-reversion speed for asset *i* (> 0).
    theta_i:
        Long-run variance level for asset *i* (> 0).
    sigma_i0_sq:
        Initial variance for asset *i* (> 0).

    Returns
    -------
    np.ndarray
        ``float64`` array of the same shape as *t*.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    assert kappa_i > 0, f"kappa_i must be positive; got {kappa_i}"
    assert theta_i > 0, f"theta_i must be positive; got {theta_i}"
    assert sigma_i0_sq > 0, f"sigma_i0_sq must be positive; got {sigma_i0_sq}"

    return theta_i + (sigma_i0_sq - theta_i) * np.exp(-kappa_i * t_arr)


# ---------------------------------------------------------------------------
# Integrand: product of expected variances
# ---------------------------------------------------------------------------


def heston_expected_det_integrand(
    t: float,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma0_sq: np.ndarray,
) -> float:
    """Product of three expected variance terms at a single time *t*.

    .. math::

        f(t) = \\prod_{i=1}^{3} E[\\sigma_{i,t}^2]

    This is the integrand of :func:`heston_expected_gen_variance` before
    scaling by ``|C| / T``.

    Parameters
    ----------
    t:
        Time point in years (≥ 0).
    kappa:
        Mean-reversion speeds, shape ``(3,)``.
    theta:
        Long-run variance levels, shape ``(3,)``.
    sigma0_sq:
        Initial variances, shape ``(3,)``.

    Returns
    -------
    float
        Scalar value of the integrand at time *t*.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma0_sq = np.asarray(sigma0_sq, dtype=np.float64)

    assert kappa.shape == (3,), f"kappa must have shape (3,); got {kappa.shape}"
    assert theta.shape == (3,), f"theta must have shape (3,); got {theta.shape}"
    assert sigma0_sq.shape == (3,), (
        f"sigma0_sq must have shape (3,); got {sigma0_sq.shape}"
    )

    ev = theta + (sigma0_sq - theta) * np.exp(-kappa * float(t))
    return float(ev[0] * ev[1] * ev[2])


# ---------------------------------------------------------------------------
# Analytic expected GRV (8-term expansion)
# ---------------------------------------------------------------------------


def heston_expected_gen_variance(
    T_array: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma0_sq: np.ndarray,
    corr_det: float,
) -> np.ndarray:
    """Vectorized expected generalized realized variance using an analytic formula.

    The integrand :math:`\\prod_i (a_i + b_i e^{-\\kappa_i t})`, where
    :math:`a_i = \\theta_i` and :math:`b_i = \\sigma_{i,0}^2 - \\theta_i`,
    is expanded into 8 exponential terms indexed by the subset
    :math:`S \\subseteq \\{1,2,3\\}` of assets that contribute the *b* factor:

    +----+-------+----------------------------------+-----------------------------+
    | k  | S     | coefficient :math:`C_k`          | rate :math:`r_k`            |
    +====+=======+==================================+=============================+
    | 1  | {}    | :math:`a_1 a_2 a_3`              | 0                           |
    | 2  | {1}   | :math:`b_1 a_2 a_3`              | :math:`\\kappa_1`           |
    | 3  | {2}   | :math:`a_1 b_2 a_3`              | :math:`\\kappa_2`           |
    | 4  | {3}   | :math:`a_1 a_2 b_3`              | :math:`\\kappa_3`           |
    | 5  | {1,2} | :math:`b_1 b_2 a_3`              | :math:`\\kappa_1+\\kappa_2` |
    | 6  | {1,3} | :math:`b_1 a_2 b_3`              | :math:`\\kappa_1+\\kappa_3` |
    | 7  | {2,3} | :math:`a_1 b_2 b_3`              | :math:`\\kappa_2+\\kappa_3` |
    | 8  | {1,2,3}| :math:`b_1 b_2 b_3`             | :math:`\\kappa_1+\\kappa_2+\\kappa_3` |
    +----+-------+----------------------------------+-----------------------------+

    The integral of :math:`C_k e^{-r_k t}` from 0 to *T* is:

    .. math::

        I_k(T) = \\begin{cases}
            C_k T                            & r_k < 10^{-10} \\\\
            C_k (1 - e^{-r_k T}) / r_k      & \\text{otherwise}
        \\end{cases}

    The expected GRV is then:

    .. math::

        E[\\sigma_R^2](T) = \\frac{|C|}{T} \\sum_{k=1}^{8} I_k(T)

    Parameters
    ----------
    T_array:
        Array of horizons in years; each element must be ≥ 0.
    kappa:
        Mean-reversion speeds, shape ``(3,)``.
    theta:
        Long-run variance levels, shape ``(3,)``.
    sigma0_sq:
        Initial variances, shape ``(3,)``.
    corr_det:
        Determinant of the 3×3 correlation matrix :math:`|C|`.

    Returns
    -------
    np.ndarray
        Expected GRV values of the same shape as *T_array*, in ``float64``.
        Entries with :math:`T \\leq 10^{-15}` are set to 0.
    """
    T = np.asarray(T_array, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma0_sq = np.asarray(sigma0_sq, dtype=np.float64)

    assert kappa.shape == (3,), f"kappa must have shape (3,); got {kappa.shape}"
    assert theta.shape == (3,), f"theta must have shape (3,); got {theta.shape}"
    assert sigma0_sq.shape == (3,), (
        f"sigma0_sq must have shape (3,); got {sigma0_sq.shape}"
    )
    assert np.all(kappa > 0), "All kappa values must be positive."
    assert np.all(theta > 0), "All theta values must be positive."
    assert np.all(sigma0_sq > 0), "All sigma0_sq values must be positive."

    a = theta               # shape (3,)
    b = sigma0_sq - theta   # shape (3,)
    k = kappa               # shape (3,)

    # 8-term expansion: (coefficient, exponential decay rate)
    # Ordered to match the table in the docstring.
    coeffs = np.array(
        [
            a[0] * a[1] * a[2],           # S = {}
            b[0] * a[1] * a[2],           # S = {1}
            a[0] * b[1] * a[2],           # S = {2}
            a[0] * a[1] * b[2],           # S = {3}
            b[0] * b[1] * a[2],           # S = {1,2}
            b[0] * a[1] * b[2],           # S = {1,3}
            a[0] * b[1] * b[2],           # S = {2,3}
            b[0] * b[1] * b[2],           # S = {1,2,3}
        ],
        dtype=np.float64,
    )

    rates = np.array(
        [
            0.0,                           # S = {}
            k[0],                          # S = {1}
            k[1],                          # S = {2}
            k[2],                          # S = {3}
            k[0] + k[1],                   # S = {1,2}
            k[0] + k[2],                   # S = {1,3}
            k[1] + k[2],                   # S = {2,3}
            k[0] + k[1] + k[2],            # S = {1,2,3}
        ],
        dtype=np.float64,
    )

    # Broadcasting shapes: coeffs/rates → (8, 1),  T → (1, n_T)
    C = coeffs[:, np.newaxis]  # (8, 1)
    R = rates[:, np.newaxis]   # (8, 1)
    T_row = T[np.newaxis, :]   # (1, n_T)

    # Analytic integral of C*exp(-R*t) from 0 to T
    # Linear limit when rate ≈ 0; exponential formula otherwise.
    use_linear = R < 1e-10  # (8, 1), broadcast to (8, n_T)

    with np.errstate(divide="ignore", invalid="ignore"):
        I_exp = C * (1.0 - np.exp(-R * T_row)) / R   # (8, n_T)
    I_lin = C * T_row                                 # (8, n_T)

    I = np.where(use_linear, I_lin, I_exp)            # (8, n_T)
    total_integral = I.sum(axis=0)                    # (n_T,)

    # E[sigma_R^2](T) = |C| * total_integral / T ; guard against T = 0
    T_safe = np.where(T > 1e-15, T, 1.0)
    result = np.where(
        T > 1e-15,
        corr_det * total_integral / T_safe,
        0.0,
    )
    return result


# ---------------------------------------------------------------------------
# NLS residuals
# ---------------------------------------------------------------------------


def heston_residuals(
    params: np.ndarray,
    T_array: np.ndarray,
    y_observed: np.ndarray,
    corr_det: float,
    sigma0_sq_fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Residuals vector for ``scipy.optimize.least_squares``.

    When ``sigma0_sq_fixed`` is supplied the initial variances are pinned to
    those values and the free parameters are only ``[kappa, theta]`` (6 total).
    This produces a genuine term-structure curve rather than a degenerate flat
    constant (which arises when ``sigma0 = theta``).

    Parameters
    ----------
    params:
        Parameter vector ``[kappa1, kappa2, kappa3, theta1, theta2, theta3]``.
    T_array:
        Horizon array, shape ``(n,)``.
    y_observed:
        Observed determinant values, shape ``(n,)``.
    corr_det:
        Determinant of the correlation matrix :math:`|C|`.
    sigma0_sq_fixed:
        If given, fix initial variances to these values instead of setting
        ``sigma0_sq = theta`` (the old reduced model).  Shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Residual vector ``y_model - y_observed``, shape ``(n,)``.
    """
    params = np.asarray(params, dtype=np.float64)
    assert params.shape == (6,), f"params must have shape (6,); got {params.shape}"

    kappa = params[:3]
    theta = params[3:6]
    if sigma0_sq_fixed is not None:
        sigma0_sq = np.asarray(sigma0_sq_fixed, dtype=np.float64)
    else:
        # Legacy reduced model: sigma_i0^2 = theta_i  (flat curve)
        sigma0_sq = theta.copy()

    y_model = heston_expected_gen_variance(
        T_array, kappa, theta, sigma0_sq, corr_det
    )
    return y_model - np.asarray(y_observed, dtype=np.float64)


# ---------------------------------------------------------------------------
# Multi-start NLS calibration
# ---------------------------------------------------------------------------


def calibrate_heston(
    det_series: pd.Series,
    T_array: np.ndarray,
    corr_matrix: np.ndarray,
    n_starts: int = 10,
    seed: int = 42,
    sigma0_sq_fixed: np.ndarray | None = None,
) -> dict:
    """Multi-start NLS calibration of the Heston model to empirical determinants.

    The parameter vector is ``[kappa1, kappa2, kappa3, theta1, theta2, theta3]``
    with bounds:

    * ``kappa_i ∈ (1e-4, 50)``
    * ``theta_i ∈ (1e-12, 1.0)``

    Initial values are drawn from:

    * ``kappa_i ~ Uniform(0.05, 2)``
    * ``theta_i ~ Uniform(1e-8, max_det / |C|)``

    ``sigma0_sq`` is fixed to ``theta`` (reduced model).

    Parameters
    ----------
    det_series:
        Empirical determinant series from
        :func:`~src.covariance.rolling_covariance_determinants` or similar,
        with a date index.
    T_array:
        Horizon array aligned to *det_series*, shape ``(n,)``.
    corr_matrix:
        3×3 full-sample correlation matrix.
    n_starts:
        Number of random restarts; defaults to 10.
    seed:
        Seed for the random number generator; defaults to 42.

    Returns
    -------
    dict
        Keys:

        * ``params`` – best-fit parameter vector ``(6,)``.
        * ``kappa`` – ``params[:3]``.
        * ``theta`` – ``params[3:6]``.
        * ``sigma0_sq`` – ``theta`` (reduced model).
        * ``fitted_values`` – model predictions at *T_array*.
        * ``residuals`` – ``fitted_values - y_observed``.
        * ``cost`` – final sum of squared residuals / 2 (``res.cost``).
        * ``jacobian`` – Jacobian matrix at the solution, shape ``(n, 6)``.
        * ``success`` – whether any optimisation run succeeded.
        * ``T_array`` – the input *T_array*.
        * ``y_observed`` – ``det_series.values``.
        * ``corr_det`` – ``np.linalg.det(corr_matrix)``.

    Raises
    ------
    ValueError
        If *corr_matrix* is not a 3×3 array or its determinant is ≤ 0.
    RuntimeError
        If all optimisation starts fail (``success=False`` for every run).
    """
    corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
    if corr_matrix.shape != (3, 3):
        raise ValueError(
            f"corr_matrix must have shape (3, 3); got {corr_matrix.shape}."
        )

    corr_det = float(np.linalg.det(corr_matrix))
    if corr_det <= 0:
        raise ValueError(
            f"Determinant of corr_matrix must be positive; got {corr_det:.6g}."
        )

    y_observed = np.asarray(det_series.values, dtype=np.float64)
    T_array = np.asarray(T_array, dtype=np.float64)

    if y_observed.shape != T_array.shape:
        raise ValueError(
            f"det_series ({y_observed.shape}) and T_array ({T_array.shape}) "
            f"must have the same length."
        )

    # Parameter bounds for least_squares (trf method).
    lb = [1e-4,  1e-4,  1e-4,  1e-12, 1e-12, 1e-12]
    ub = [50.0,  50.0,  50.0,  1.0,   1.0,   1.0  ]
    bounds = (lb, ub)

    # Upper bound for theta initial draws.
    max_det = float(np.max(np.abs(y_observed)))
    theta_draw_upper = float(
        np.clip((max_det / corr_det) ** (1.0 / 3.0), 1e-8, 1.0)
    )

    rng = np.random.default_rng(seed)
    best_result = None
    best_cost = np.inf

    for i in range(n_starts):
        kappa0 = rng.uniform(0.05, 2.0, size=3)
        theta0 = rng.uniform(1e-8, theta_draw_upper, size=3)
        # Ensure initial point is strictly inside bounds.
        kappa0 = np.clip(kappa0, lb[0], ub[0])
        theta0 = np.clip(theta0, lb[3], ub[3])
        p0 = np.concatenate([kappa0, theta0])

        try:
            res = least_squares(
                heston_residuals,
                p0,
                args=(T_array, y_observed, corr_det, sigma0_sq_fixed),
                method="trf",
                bounds=bounds,
                max_nfev=10_000,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Start %d failed with exception: %s", i, exc)
            continue

        logger.debug(
            "Start %d/%d  cost=%.6g  success=%s",
            i + 1,
            n_starts,
            res.cost,
            res.success,
        )

        if res.cost < best_cost:
            best_cost = res.cost
            best_result = res

    if best_result is None:
        raise RuntimeError(
            "All optimisation starts failed.  "
            "Check that det_series contains valid (positive) values."
        )

    best_params = best_result.x.astype(np.float64)
    kappa_fit = best_params[:3]
    theta_fit = best_params[3:6]
    # Use first-window sigma0 when provided; fall back to reduced-model (flat) otherwise
    sigma0_sq_fit = (
        np.asarray(sigma0_sq_fixed, dtype=np.float64)
        if sigma0_sq_fixed is not None
        else theta_fit.copy()
    )

    fitted_values = heston_expected_gen_variance(
        T_array, kappa_fit, theta_fit, sigma0_sq_fit, corr_det
    )
    residuals = fitted_values - y_observed

    logger.info(
        "calibrate_heston: best cost=%.6g  success=%s",
        best_result.cost,
        best_result.success,
    )

    return {
        "params":        best_params,
        "kappa":         kappa_fit,
        "theta":         theta_fit,
        "sigma0_sq":     sigma0_sq_fit,
        "fitted_values": fitted_values,
        "residuals":     residuals,
        "cost":          float(best_result.cost),
        "jacobian":      best_result.jac,
        "success":       bool(best_result.success),
        "T_array":       T_array,
        "y_observed":    y_observed,
        "corr_det":      corr_det,
    }


# ---------------------------------------------------------------------------
# Frequentist inference from Jacobian
# ---------------------------------------------------------------------------


def heston_inference(calib_result: dict) -> dict:
    """Compute frequentist inference statistics from a calibration result.

    Standard errors, *t*-values, and *p*-values are derived from the Jacobian
    returned by ``scipy.optimize.least_squares`` using the standard nonlinear
    least-squares sandwich formula:

    .. math::

        \\widehat{\\operatorname{Cov}}(\\hat{\\theta}) =
            s^2 (J^\\top J)^{-1}, \\quad
            s^2 = \\frac{RSS}{n - p}

    where the matrix inverse is replaced by the Moore–Penrose pseudoinverse
    when :math:`J^\\top J` is numerically singular.

    Parameters
    ----------
    calib_result:
        Dictionary returned by :func:`calibrate_heston`.

    Returns
    -------
    dict
        Keys:

        * ``param_names`` – list of parameter name strings.
        * ``params`` – fitted parameter values, shape ``(6,)``.
        * ``std_errors`` – standard errors, shape ``(6,)``.
        * ``t_values`` – *t*-statistics ``params / std_errors``, shape ``(6,)``.
        * ``p_values`` – two-sided *p*-values from the *t*-distribution with
          ``n - p`` degrees of freedom, shape ``(6,)``.

    Raises
    ------
    ValueError
        If the Jacobian stored in *calib_result* is ``None`` or has fewer
        rows than parameters.
    """
    J: np.ndarray | None = calib_result.get("jacobian")
    if J is None:
        raise ValueError(
            "calib_result['jacobian'] is None; "
            "the optimiser did not return a Jacobian."
        )

    J = np.asarray(J, dtype=np.float64)
    params = np.asarray(calib_result["params"], dtype=np.float64)
    residuals = np.asarray(calib_result["residuals"], dtype=np.float64)
    y_observed = np.asarray(calib_result["y_observed"], dtype=np.float64)

    n = len(y_observed)
    p = len(params)

    if J.shape[0] < p:
        raise ValueError(
            f"Jacobian has {J.shape[0]} rows but there are {p} parameters; "
            f"cannot compute standard errors."
        )
    if n <= p:
        raise ValueError(
            f"Degrees of freedom = {n - p} ≤ 0; inference is not possible "
            f"(n={n}, p={p})."
        )

    RSS = float(np.dot(residuals, residuals))
    s_sq = RSS / (n - p)

    JtJ = J.T @ J  # (p, p)
    try:
        cov_params = s_sq * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        logger.warning(
            "J^T J is singular; using Moore–Penrose pseudoinverse for "
            "covariance estimation."
        )
        cov_params = s_sq * np.linalg.pinv(JtJ)

    # Guard against negative diagonal entries due to numerical issues.
    diag = np.maximum(np.diag(cov_params), 0.0)
    std_errors = np.sqrt(diag)

    # Avoid division by zero in t-values.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(std_errors > 0, params / std_errors, np.nan)

    df = float(n - p)
    p_values = np.where(
        np.isfinite(t_values),
        2.0 * (1.0 - t_dist.cdf(np.abs(t_values), df=df)),
        np.nan,
    )

    param_names = [
        "kappa1", "kappa2", "kappa3",
        "theta1", "theta2", "theta3",
    ]

    return {
        "param_names": param_names,
        "params":      params,
        "std_errors":  std_errors,
        "t_values":    t_values,
        "p_values":    p_values,
    }
