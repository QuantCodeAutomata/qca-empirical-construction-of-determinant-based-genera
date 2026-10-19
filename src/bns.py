"""Barndorff-Nielsen-Shephard (BNS) three-asset model for generalized realized variance.

The BNS model drives each asset's variance process via a Lévy subordinator:
    d sigma_i^2 = -lambda * sigma_i^2 dt + dZ^(i)_{lambda*t}

Expected variance:
    m_i(t) = E[sigma_i_t^2] = exp(-lambda*t)*(sigma_i0_sq - kappa_i1) + kappa_i1

Variance of variance:
    Var[sigma_i_t^2] = (kappa_i2/2)*(1 - exp(-2*lambda*t))

Jump-inclusive covariance rate:
    SigmaTilde_t^B = Sigma_t^{B,c} + lambda * kappa_{0,2} * rho * rho^T
where Sigma_t^{B,c} = D_t * R^B * D_t, D_t = diag(sigma_1t, sigma_2t, sigma_3t).

Determinant expansion (matrix determinant lemma, dropping inconsistent 1/2 factor):
    |SigmaTilde_t^B| = |Sigma_t^{B,c}| * (1 + lambda*kappa_{0,2}*rho^T*(Sigma_t^{B,c})^{-1}*rho)

Expected generalized realized variance (Theorem 4.1):
    E[sigma_R^2](T) = |R^B|/T * (E0 + lambda*kappa02*total_correction)
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core BNS moment functions
# ---------------------------------------------------------------------------

def bns_mean_variance(
    t: np.ndarray,
    lam: float,
    kappa_i1: float,
    sigma_i0_sq: float,
) -> np.ndarray:
    """Compute E[sigma_i_t^2] under the BNS model.

    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate the conditional mean.
    lam : float
        Mean-reversion speed lambda > 0.
    kappa_i1 : float
        Long-run mean level of sigma_i^2 (first cumulant of the BDLP).
    sigma_i0_sq : float
        Initial variance sigma_i0^2 = sigma_i^2(0).

    Returns
    -------
    np.ndarray
        m_i(t) = exp(-lam*t)*(sigma_i0_sq - kappa_i1) + kappa_i1, same shape as t.
    """
    t = np.asarray(t, dtype=float)
    return np.exp(-lam * t) * (sigma_i0_sq - kappa_i1) + kappa_i1


def bns_var_variance(
    t: np.ndarray,
    lam: float,
    kappa_i2: float,
) -> np.ndarray:
    """Compute Var[sigma_i_t^2] under the BNS model.

    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate the conditional variance.
    lam : float
        Mean-reversion speed lambda > 0.
    kappa_i2 : float
        Second cumulant of the BDLP Lévy measure.

    Returns
    -------
    np.ndarray
        (kappa_i2/2)*(1 - exp(-2*lam*t)), same shape as t.
    """
    t = np.asarray(t, dtype=float)
    return (kappa_i2 / 2.0) * (1.0 - np.exp(-2.0 * lam * t))


def bns_expected_sigma(
    t: np.ndarray,
    lam: float,
    kappa_i1: float,
    sigma_i0_sq: float,
    kappa_i2: float,
) -> np.ndarray:
    """Brockhaus-Long approximation for E[sigma_i_t].

    Approximates the expected volatility (not variance) using a second-order
    Taylor expansion around the mean variance m_i(t).

    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate the expected volatility.
    lam : float
        Mean-reversion speed lambda > 0.
    kappa_i1 : float
        Long-run mean level of sigma_i^2.
    sigma_i0_sq : float
        Initial variance sigma_i0^2.
    kappa_i2 : float
        Second cumulant of the BDLP Lévy measure.

    Returns
    -------
    np.ndarray
        E[sigma_i_t] ≈ sqrt(m_i(t)) - kappa_i2*(1-exp(-2*lam*t)) / (16*m_i(t)^(3/2)).
        m_i(t) is clipped to 1e-20 to avoid division by zero.
    """
    t = np.asarray(t, dtype=float)
    m = bns_mean_variance(t, lam, kappa_i1, sigma_i0_sq)
    m_safe = np.clip(m, 1e-20, None)
    correction = kappa_i2 * (1.0 - np.exp(-2.0 * lam * t)) / (16.0 * m_safe ** 1.5)
    return np.sqrt(m_safe) - correction


# ---------------------------------------------------------------------------
# Analytic and numerical integral routines
# ---------------------------------------------------------------------------

def bns_product_mean_integral(
    T: float,
    lam: float,
    kappa1: np.ndarray,
    sigma0_sq: np.ndarray,
) -> float:
    """Analytic integral of prod_i m_i(t) over [0, T] for three assets.

    Expands the product of three BNS mean functions into 2^3 = 8 terms using
    the binary subset enumeration:
        prod_i (a_i + b_i*exp(-lam*t))  where a_i=kappa_i1, b_i=sigma_i0_sq_i - kappa_i1.

    Each term coeff * exp(-n*lam*t) integrates to:
        coeff * T                               if n = 0
        coeff * (1 - exp(-n*lam*T)) / (n*lam)  if n > 0

    Parameters
    ----------
    T : float
        Upper limit of integration (maturity).
    lam : float
        Common mean-reversion speed lambda > 0.
    kappa1 : np.ndarray
        Long-run mean levels [kappa_11, kappa_21, kappa_31], shape (3,).
    sigma0_sq : np.ndarray
        Initial variances [sigma_10^2, sigma_20^2, sigma_30^2], shape (3,).

    Returns
    -------
    float
        integral_0^T  prod_{i=1}^{3} m_i(t)  dt.
    """
    a = np.asarray(kappa1, dtype=float)
    b = np.asarray(sigma0_sq, dtype=float) - a   # b_i = sigma_i0_sq - kappa_i1

    total = 0.0
    for s in range(8):   # iterate over all 2^3 binary subsets
        coeff = 1.0
        n = 0            # number of b_k factors (= number of exp(-lam*t) factors)
        for k in range(3):
            if (s >> k) & 1:
                coeff *= b[k]
                n += 1
            else:
                coeff *= a[k]

        if n == 0:
            total += coeff * T
        else:
            total += coeff * (1.0 - np.exp(-n * lam * T)) / (n * lam)

    return total


def bns_off_diagonal_integral(
    T: float,
    lam: float,
    kappa_i: np.ndarray,
    sigma0_sq: np.ndarray,
    kappa_i2: np.ndarray,
    i_idx: int,
    j_idx: int,
) -> float:
    """Numerically integrate E[sigma_i_t]*E[sigma_j_t]*m_k(t) over [0, T].

    Used for the off-diagonal correction terms in the expected generalized
    realized variance formula. The Brockhaus-Long approximation is used for
    E[sigma_i_t] and E[sigma_j_t]; the remaining factor m_k(t) is the exact
    BNS conditional mean. The three variances are independent under BNS.

    Parameters
    ----------
    T : float
        Upper limit of integration.
    lam : float
        Common mean-reversion speed lambda > 0.
    kappa_i : np.ndarray
        Long-run mean levels [kappa_11, kappa_21, kappa_31], shape (3,).
    sigma0_sq : np.ndarray
        Initial variances, shape (3,).
    kappa_i2 : np.ndarray
        Second cumulants [kappa_12, kappa_22, kappa_32], shape (3,).
    i_idx : int
        Index of first asset in {0, 1, 2}.
    j_idx : int
        Index of second asset in {0, 1, 2}, must differ from i_idx.

    Returns
    -------
    float
        integral_0^T  E[sigma_{i}_t] * E[sigma_{j}_t] * m_{k}(t)  dt,
        where k is the remaining index in {0,1,2} minus {i_idx, j_idx}.
    """
    k_idx = ({0, 1, 2} - {i_idx, j_idx}).pop()

    # Pre-extract scalars so the inner loop avoids array indexing overhead.
    a_i, b_i, v_i = float(kappa_i[i_idx]), float(sigma0_sq[i_idx] - kappa_i[i_idx]), float(kappa_i2[i_idx])
    a_j, b_j, v_j = float(kappa_i[j_idx]), float(sigma0_sq[j_idx] - kappa_i[j_idx]), float(kappa_i2[j_idx])
    a_k, b_k     = float(kappa_i[k_idx]), float(sigma0_sq[k_idx] - kappa_i[k_idx])

    def integrand(t: float) -> float:
        """Compute E[sigma_i_t] * E[sigma_j_t] * m_k(t) using inline scalar ops."""
        exp_lt  = np.exp(-lam * t)
        exp_2lt = exp_lt * exp_lt

        # Brockhaus-Long E[sigma_i_t]
        m_i    = b_i * exp_lt + a_i
        m_i_s  = m_i if m_i > 1e-20 else 1e-20
        E_i    = m_i_s ** 0.5 - v_i * (1.0 - exp_2lt) / (16.0 * m_i_s ** 1.5)

        # Brockhaus-Long E[sigma_j_t]
        m_j    = b_j * exp_lt + a_j
        m_j_s  = m_j if m_j > 1e-20 else 1e-20
        E_j    = m_j_s ** 0.5 - v_j * (1.0 - exp_2lt) / (16.0 * m_j_s ** 1.5)

        # Exact BNS mean variance m_k(t)
        m_k    = b_k * exp_lt + a_k

        return E_i * E_j * m_k

    result, _ = quad(integrand, 0.0, T, limit=50, epsabs=1e-10, epsrel=1e-8)
    return result


# ---------------------------------------------------------------------------
# Expected generalized realized variance
# ---------------------------------------------------------------------------

def _batch_gl_quadrature(
    integrand_fn,
    T_array: np.ndarray,
    n_points: int = 20,
    _gl_cache: dict = {},
) -> np.ndarray:
    """Vectorized Gauss-Legendre quadrature: integral_0^T f(t) dt for all T in T_array.

    Transforms [0, T] -> [-1, 1] via t = T/2*(x+1) then applies GL weights.
    The integrand_fn must accept a 2-D array of shape (n_valid, n_points) and
    return a 2-D array of the same shape (fully vectorized).

    GL nodes/weights are cached by n_points to avoid recomputing on every call.
    """
    if n_points not in _gl_cache:
        _gl_cache[n_points] = np.polynomial.legendre.leggauss(n_points)
    x, w = _gl_cache[n_points]
    T = np.asarray(T_array, dtype=float)
    valid = T > 1e-10
    result = np.zeros(len(T))
    if not valid.any():
        return result
    T_v = T[valid]
    # Transform: t = T/2*(x+1), dt = T/2*dx
    t_nodes = (T_v[:, None] / 2.0) * (x[None, :] + 1.0)  # (n_valid, n_points)
    weights = (T_v[:, None] / 2.0) * w[None, :]            # (n_valid, n_points)
    values = integrand_fn(t_nodes)                          # (n_valid, n_points)
    result[valid] = np.sum(weights * values, axis=1)
    return result


def _bns_product_mean_integral_vec(
    T_array: np.ndarray,
    lam: float,
    kappa1: np.ndarray,
    sigma0_sq: np.ndarray,
) -> np.ndarray:
    """Vectorized analytic integral of prod_i m_i(t) over [0,T] for all T in T_array."""
    a = np.asarray(kappa1, dtype=float)
    b = np.asarray(sigma0_sq, dtype=float) - a
    T = np.asarray(T_array, dtype=float)  # (N,)
    total = np.zeros(len(T))
    for s in range(8):
        coeff = 1.0
        n = 0
        for k_bit in range(3):
            if (s >> k_bit) & 1:
                coeff *= b[k_bit]; n += 1
            else:
                coeff *= a[k_bit]
        if n == 0:
            total += coeff * T
        else:
            total += coeff * (1.0 - np.exp(-n * lam * T)) / (n * lam)
    return total


def _bns_pairwise_integral_analytic(
    T_array: np.ndarray,
    lam: float,
    a_j: float,
    b_j: float,
    a_k: float,
    b_k: float,
) -> np.ndarray:
    """Analytic integral of (a_j + b_j*exp(-lam*t))*(a_k + b_k*exp(-lam*t)) over [0,T].

    Expands to four terms:
        a_j*a_k * T
        + (a_j*b_k + a_k*b_j) * (1 - exp(-lam*T)) / lam
        + b_j*b_k * (1 - exp(-2*lam*T)) / (2*lam)

    Used for the diagonal correction in the BNS determinant expansion, where
    each diagonal term i requires the integral of m_j(t)*m_k(t) (the product
    of the *other* two asset mean-variance functions), NOT all three.

    Parameters
    ----------
    T_array : np.ndarray  -- upper limits of integration
    lam     : float       -- common mean-reversion speed
    a_j, b_j : float      -- kappa1_j and (sigma0_sq_j - kappa1_j)
    a_k, b_k : float      -- kappa1_k and (sigma0_sq_k - kappa1_k)

    Returns
    -------
    np.ndarray  -- integral value for each T in T_array
    """
    T = np.asarray(T_array, dtype=float)
    valid = T > 1e-10
    result = np.zeros(len(T))
    if not valid.any():
        return result
    T_v = T[valid]
    e1 = np.exp(-lam * T_v)
    e2 = np.exp(-2.0 * lam * T_v)
    result[valid] = (
        a_j * a_k * T_v
        + (a_j * b_k + a_k * b_j) * (1.0 - e1) / lam
        + b_j * b_k * (1.0 - e2) / (2.0 * lam)
    )
    return result


def bns_expected_gen_variance(
    T_array: np.ndarray,
    lam: float,
    kappa1: np.ndarray,
    sigma0_sq: np.ndarray,
    kappa_i2: np.ndarray,
    kappa02: float,
    rho: np.ndarray,
    RB: np.ndarray,
    n_gl: int = 20,
    _RB_det: float = None,
    _delta: np.ndarray = None,
) -> np.ndarray:
    """Compute E[sigma_R^2](T) for each T under the three-asset BNS model.

    Uses vectorised batch Gauss-Legendre quadrature (n_gl points) for the
    off-diagonal correction integrals so that the entire T_array is processed
    in a single pass without per-observation Python loops.

    Parameters
    ----------
    T_array : np.ndarray  -- time horizons
    lam : float           -- mean-reversion speed
    kappa1 : np.ndarray   -- long-run mean levels, shape (3,)
    sigma0_sq : np.ndarray -- initial variances, shape (3,)
    kappa_i2 : np.ndarray  -- second cumulants, shape (3,)
    kappa02 : float        -- jump amplitude
    rho : np.ndarray       -- jump loading vector, shape (3,)
    RB : np.ndarray        -- correlation matrix, shape (3,3)
    n_gl : int             -- number of GL quadrature points (default 20)
    _RB_det : float, optional -- pre-computed det(RB); computed from RB if None
    _delta  : np.ndarray, optional -- pre-computed inv(RB); computed from RB if None

    Returns
    -------
    np.ndarray
        E[sigma_R^2](T) for each T; 0 where T <= 1e-10.
    """
    kappa1 = np.asarray(kappa1, dtype=float)
    sigma0_sq = np.asarray(sigma0_sq, dtype=float)
    kappa_i2 = np.asarray(kappa_i2, dtype=float)
    rho = np.asarray(rho, dtype=float)
    RB_det = _RB_det if _RB_det is not None else float(np.linalg.det(RB))
    delta = _delta if _delta is not None else np.linalg.inv(RB)

    # Off-diagonal pairs (i < j) and their coefficients 2*delta[i,j]*rho[i]*rho[j]
    off_diag_pairs = [
        (i, j, 2.0 * delta[i, j] * rho[i] * rho[j])
        for i in range(3)
        for j in range(i + 1, 3)
    ]

    T_array = np.asarray(T_array, dtype=float)
    valid = T_array > 1e-10

    # E0: analytic integral for all T at once
    E0 = _bns_product_mean_integral_vec(T_array, lam, kappa1, sigma0_sq)

    # Diagonal correction (correct formula, Bug fix):
    #   sum_i delta[i,i]*rho[i]^2 * integral_0^T m_j(t)*m_k(t) dt   (j,k != i)
    #
    # Derivation: from the matrix determinant lemma,
    #   det(SigmaTilde) = det(Sigma^c) * (1 + lam*kappa02 * rho'*(Sigma^c)^{-1}*rho)
    # Expanding rho'*(Sigma^c)^{-1}*rho = sum_i delta[i,i]*rho[i]^2/sigma[i]^2 + cross
    # Multiplying by det(Sigma^c)=|R^B|*prod sigma_i^2 and taking E[.]:
    #   diagonal term for asset i = delta[i,i]*rho[i]^2 * prod_{k!=i} E[sigma_k^2]
    #                             = delta[i,i]*rho[i]^2 * m_j(t) * m_k(t)
    # This is a product of TWO m_i values, NOT three.  The old code incorrectly
    # used diag_coeff * E0 = (sum delta[i,i]*rho[i]^2) * integral(prod_k m_k),
    # which included an extra factor of m_i(t) and was ~1000× too small at
    # daily variance scale (~1e-3), letting negative off-diagonal terms dominate.
    diag_correction = np.zeros(len(T_array))
    for i in range(3):
        others = [x for x in range(3) if x != i]
        j_idx, k_idx = others[0], others[1]
        a_j = float(kappa1[j_idx]);  b_j = float(sigma0_sq[j_idx] - kappa1[j_idx])
        a_k = float(kappa1[k_idx]);  b_k = float(sigma0_sq[k_idx] - kappa1[k_idx])
        diag_correction += (
            delta[i, i] * rho[i] ** 2
            * _bns_pairwise_integral_analytic(T_array, lam, a_j, b_j, a_k, b_k)
        )

    # Off-diagonal corrections via batch GL quadrature
    off_correction = np.zeros(len(T_array))
    for i_idx, j_idx, coeff_ij in off_diag_pairs:
        k_idx = ({0, 1, 2} - {i_idx, j_idx}).pop()
        # Pre-extract scalars to avoid repeated indexing inside lambda
        a_i = float(kappa1[i_idx]);  b_i = float(sigma0_sq[i_idx] - kappa1[i_idx]);  v_i = float(kappa_i2[i_idx])
        a_j = float(kappa1[j_idx]);  b_j = float(sigma0_sq[j_idx] - kappa1[j_idx]);  v_j = float(kappa_i2[j_idx])
        a_k = float(kappa1[k_idx]);  b_k = float(sigma0_sq[k_idx] - kappa1[k_idx])

        def _integrand(t, _a_i=a_i, _b_i=b_i, _v_i=v_i,
                           _a_j=a_j, _b_j=b_j, _v_j=v_j,
                           _a_k=a_k, _b_k=b_k):
            """Vectorized integrand: E[sigma_i_t]*E[sigma_j_t]*m_k(t), t is 2-D."""
            exp_lt  = np.exp(-lam * t)
            exp_2lt = exp_lt * exp_lt
            m_i = np.clip(_b_i * exp_lt + _a_i, 1e-20, None)
            E_i = np.sqrt(m_i) - _v_i * (1.0 - exp_2lt) / (16.0 * m_i ** 1.5)
            m_j = np.clip(_b_j * exp_lt + _a_j, 1e-20, None)
            E_j = np.sqrt(m_j) - _v_j * (1.0 - exp_2lt) / (16.0 * m_j ** 1.5)
            m_k = _b_k * exp_lt + _a_k
            return E_i * E_j * m_k

        off_correction += coeff_ij * _batch_gl_quadrature(_integrand, T_array, n_gl)

    total_correction = diag_correction + off_correction

    result = np.zeros(len(T_array))
    result[valid] = (
        RB_det / T_array[valid] * (E0[valid] + lam * kappa02 * total_correction[valid])
    )
    return result


# ---------------------------------------------------------------------------
# NLS calibration
# ---------------------------------------------------------------------------

def bns_residuals(
    params: np.ndarray,
    T_array: np.ndarray,
    y_observed: np.ndarray,
    RB: np.ndarray,
    rho: np.ndarray,
    kappa_i2_fixed: np.ndarray,
    _RB_det: float = None,
    _delta: np.ndarray = None,
    _n_gl: int = 20,
) -> np.ndarray:
    """Residuals (fitted minus observed) for BNS NLS calibration.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector of length 8:
        [lam, kappa11, kappa12, kappa13, sigma10_sq, sigma20_sq, sigma30_sq, kappa02].
    T_array : np.ndarray
        Array of time horizons.
    y_observed : np.ndarray
        Observed generalized realized variance values.
    RB : np.ndarray
        Correlation matrix R^B, shape (3, 3).
    rho : np.ndarray
        Jump loading vector, shape (3,).
    kappa_i2_fixed : np.ndarray
        Fixed second cumulants, shape (3,).
    _RB_det : float, optional
        Pre-computed det(RB) to avoid recomputation on every call.
    _delta : np.ndarray, optional
        Pre-computed inv(RB) to avoid recomputation on every call.
    _n_gl : int
        Number of GL quadrature points to use during optimization.

    Returns
    -------
    np.ndarray
        fitted - observed, length equal to len(T_array).
    """
    lam = params[0]
    kappa1 = params[1:4]
    sigma0_sq = params[4:7]
    kappa02 = params[7]

    fitted = bns_expected_gen_variance(
        T_array, lam, kappa1, sigma0_sq, kappa_i2_fixed, kappa02, rho, RB,
        n_gl=_n_gl, _RB_det=_RB_det, _delta=_delta,
    )
    return fitted - y_observed


def calibrate_bns(
    det_series: pd.Series,
    T_array: np.ndarray,
    corr_matrix: np.ndarray,
    n_starts: int = 3,
    seed: int = 42,
    fit_stride: int = 4,
    n_gl_opt: int = 15,
    n_gl_final: int = 30,
    sigma0_sq_init: np.ndarray | None = None,
) -> dict:
    """Multi-start NLS calibration of the BNS model to observed generalized variance.

    Fixes the jump loading vector rho = [1,1,1]/sqrt(3) and the second cumulants
    kappa_i2 ≈ 0 (set to 1e-8 each to avoid Brockhaus-Long correction overflow at
    daily-return scale), then estimates 8 free parameters via nonlinear least squares
    (trust-region reflective method) with multiple random restarts.

    Performance optimisations
    -------------------------
    * det(RB) and inv(RB) are computed once before the optimization loop and
      passed to the residual function, avoiding ~1000 redundant 3x3 ops per call.
    * GL nodes/weights are cached in the module-level dict inside
      _batch_gl_quadrature.
    * For optimization, a strided subsample of every *fit_stride*-th observation
      is used (default stride=4 ≈ 250 pts from ~996), cutting per-call time by
      the same factor while preserving overall curve shape.  Final residuals and
      Jacobian are evaluated on the full series.
    * n_gl is kept small (default 15) during optimization and raised to 30 for
      the final evaluation so that reported metrics are more accurate.

    Free parameters
    ---------------
    lam      : mean-reversion speed, in (1e-4, 20)
    kappa_i1 : long-run mean levels for i=1,2,3, each in (1e-12, 1.0)
    sigma_i0_sq : initial variances for i=1,2,3, each in (1e-12, 1.0)
    kappa02  : jump amplitude, in (1e-12, 5.0)

    Parameters
    ----------
    det_series : pd.Series
        Observed generalized realized variance (det-based) time series,
        length must match T_array.
    T_array : np.ndarray
        Time horizons corresponding to each observation in det_series.
    corr_matrix : np.ndarray
        Empirical correlation matrix R^B, shape (3, 3).
    n_starts : int
        Number of random starting points for multi-start optimization.
    seed : int
        Random seed for reproducibility.
    fit_stride : int
        Subsample stride used during optimization (every fit_stride-th point).
    n_gl_opt : int
        Number of GL quadrature points used during optimization iterations.
    n_gl_final : int
        Number of GL quadrature points for the final fitted-value evaluation.
    sigma0_sq_init : np.ndarray or None
        If provided (shape (3,)), the warm-start sigma0_sq values for each asset.
        Use the first-window per-asset sample variances so that sigma0 ≠ kappa1
        at the start, breaking the degenerate flat-curve local minimum.

    Returns
    -------
    dict
        Keys:
          params       : np.ndarray, shape (8,), best parameter vector
          lam          : float
          kappa1       : np.ndarray, shape (3,)
          sigma0_sq    : np.ndarray, shape (3,)
          kappa02      : float
          fitted_values: np.ndarray, shape (len(T_array),), evaluated on full series
          residuals    : np.ndarray, fitted - observed, full series
          cost         : float, 0.5 * sum(residuals^2) from scipy (subsample)
          jacobian     : np.ndarray, Jacobian at solution (subsample)
          success      : bool

    Raises
    ------
    RuntimeError
        If all calibration starts fail.
    """
    rho = np.ones(3) / np.sqrt(3)
    # kappa_i2 controls variance-of-variance; 0.1 is ~10^7× too large for daily
    # returns (m_i ≈ 2e-4), making the Brockhaus-Long correction >> sqrt(m_i)
    # and producing negative E[sigma_i] values.  Set effectively to zero.
    kappa_i2_fixed = np.array([1e-8, 1e-8, 1e-8])
    y_observed = (
        det_series.values if isinstance(det_series, pd.Series) else np.asarray(det_series, dtype=float)
    )
    T_full = np.asarray(T_array, dtype=float)

    # Pre-compute RB constants once (amortizes cost across all NLS iterations)
    RB_det = float(np.linalg.det(corr_matrix))
    delta = np.linalg.inv(corr_matrix)

    # Subsample for optimization speed: use every fit_stride-th observation
    idx_sub = np.arange(0, len(T_full), fit_stride)
    T_sub = T_full[idx_sub]
    y_sub = y_observed[idx_sub]

    # Data-driven variance upper bound:
    #   At steady state E[det] ≈ |R^B| * prod_i kappa1_i,
    #   so kappa1_i ≈ (max_det / |R^B|)^{1/3} at the cube-root scale.
    # Searching over [0, 1] (the previous hard-coded upper) is 100-1000× too large
    # for daily return variances (~1e-5 to 5e-3), causing optimizer divergence.
    max_det = float(np.max(np.abs(y_observed)))
    mean_det = float(np.mean(y_observed[y_observed > 0])) if np.any(y_observed > 0) else max_det
    rb_det_pos = max(abs(RB_det), 1e-10)
    # upper bound per-asset variance: cube root of max observed det / |R^B|
    var_upper = float((max_det / rb_det_pos) ** (1.0 / 3.0))
    # central initialisation variance: cube root of mean observed det / |R^B|
    var_init = float((mean_det / rb_det_pos) ** (1.0 / 3.0))

    lower = [1e-4,  1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
    upper = [20.0,  var_upper, var_upper, var_upper,
                    var_upper, var_upper, var_upper,
                    var_upper * 20.0]   # kappa02 can be larger but stays data-scaled

    rng = np.random.default_rng(seed)
    best_result = None
    best_cost = np.inf

    for start_idx in range(n_starts):
        if start_idx == 0:
            # Warm start: kappa1 at cube-root-mean scale; sigma0_sq uses first-window
            # per-asset variances when provided so sigma0 ≠ kappa1 and the optimizer
            # starts with a genuine term-structure rather than a degenerate flat curve.
            if sigma0_sq_init is not None:
                s0 = np.clip(np.asarray(sigma0_sq_init, dtype=float), lower[4], upper[4])
            else:
                s0 = np.array([var_init, var_init, var_init])
            p0 = np.array([
                1.0,
                var_init, var_init, var_init,
                s0[0], s0[1], s0[2],
                var_init * 0.1,
            ])
        else:
            p0 = rng.uniform(lower, upper)
        # Clip to bounds (warm start may underflow at extremely small var_init)
        p0 = np.clip(p0, lower, upper)
        try:
            result = least_squares(
                bns_residuals,
                p0,
                args=(T_sub, y_sub, corr_matrix, rho, kappa_i2_fixed,
                      RB_det, delta, n_gl_opt),
                method='trf',
                bounds=(lower, upper),
                max_nfev=2000,
                ftol=1e-6,
                xtol=1e-6,
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                logger.debug("BNS start %d: cost=%.6e", start_idx, result.cost)
        except Exception as exc:
            logger.warning("BNS calibration start %d failed: %s", start_idx, exc)

    if best_result is None:
        raise RuntimeError("All BNS calibration starts failed.")

    params = best_result.x
    lam = float(params[0])
    kappa1 = params[1:4]
    sigma0_sq = params[4:7]
    kappa02 = float(params[7])

    # Final evaluation on the full series with higher-accuracy quadrature
    fitted_values = bns_expected_gen_variance(
        T_full, lam, kappa1, sigma0_sq, kappa_i2_fixed, kappa02, rho, corr_matrix,
        n_gl=n_gl_final, _RB_det=RB_det, _delta=delta,
    )
    residuals = fitted_values - y_observed

    return {
        'params': params,
        'lam': lam,
        'kappa1': kappa1,
        'sigma0_sq': sigma0_sq,
        'kappa02': kappa02,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'cost': best_result.cost,
        'jacobian': best_result.jac,
        'success': bool(best_result.success),
    }


def bns_inference(calib_result: dict) -> dict:
    """Compute frequentist inference statistics from BNS calibration output.

    Uses the Jacobian matrix returned by scipy.optimize.least_squares to construct
    an approximate parameter covariance matrix under the assumption of i.i.d. errors:
        Cov(params) ≈ sigma^2 * (J^T J)^{-1}
        sigma^2     = residuals^T @ residuals / (n - p)

    Then computes asymptotic t-statistics and two-sided p-values.

    Parameters
    ----------
    calib_result : dict
        Output dictionary from calibrate_bns.

    Returns
    -------
    dict
        Keys:
          params      : np.ndarray, calibrated parameters
          std_errors  : np.ndarray, parameter standard errors
          t_stats     : np.ndarray, t-statistics (params / std_errors)
          p_values    : np.ndarray, two-sided p-values from t(dof)
          param_names : list[str], human-readable parameter labels
          dof         : int, degrees of freedom n - p
          cov         : np.ndarray, approximate covariance matrix

    Raises
    ------
    ValueError
        If the number of observations is not greater than the number of parameters.
    """
    params = calib_result['params']
    residuals = calib_result['residuals']
    jac = calib_result['jacobian']

    # Jacobian may come from a subsample used during optimization (rows = n_sub),
    # while residuals are evaluated on the full series.  Use the Jacobian row count
    # for degrees-of-freedom to keep inference internally consistent with the fit.
    n = int(jac.shape[0]) if jac is not None and jac.ndim == 2 else len(residuals)
    p = len(params)
    if n <= p:
        raise ValueError(
            f"Insufficient observations ({n}) to compute inference for {p} parameters."
        )

    dof = n - p
    # sigma2: prefer 2*cost (subsample RSS) if stored, else fall back to full RSS
    cost = calib_result.get('cost', None)
    if cost is not None:
        sigma2 = float(2.0 * cost / dof)
    else:
        sigma2 = float(np.dot(residuals, residuals) / dof)

    JtJ = jac.T @ jac
    try:
        cov = sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        logger.warning("Singular Jacobian; using Moore-Penrose pseudoinverse for covariance.")
        cov = sigma2 * np.linalg.pinv(JtJ)

    std_errors = np.sqrt(np.abs(np.diag(cov)))
    t_stats = np.where(std_errors > 0.0, params / std_errors, np.nan)
    p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stats), df=dof))

    param_names = [
        'lambda',
        'kappa11', 'kappa12', 'kappa13',
        'sigma10_sq', 'sigma20_sq', 'sigma30_sq',
        'kappa02',
    ]

    return {
        'params': params,
        'std_errors': std_errors,
        't_stats': t_stats,
        't_values': t_stats,  # alias for consistency with heston_inference
        'p_values': p_values,
        'param_names': param_names,
        'dof': dof,
        'cov': cov,
    }
