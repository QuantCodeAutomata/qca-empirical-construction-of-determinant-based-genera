"""Tests for src/bns.py — BNS model functions."""
import numpy as np
import pandas as pd
import pytest
from src.bns import (
    bns_mean_variance,
    bns_var_variance,
    bns_expected_sigma,
    bns_product_mean_integral,
    bns_off_diagonal_integral,
    bns_expected_gen_variance,
    bns_inference,
    calibrate_bns,
)


class TestBnsMeanVariance:
    def test_at_zero(self):
        m = bns_mean_variance(np.array([0.0]), lam=1.0, kappa_i1=0.04, sigma_i0_sq=0.09)
        assert m[0] == pytest.approx(0.09)

    def test_long_run_convergence(self):
        m = bns_mean_variance(np.array([1000.0]), lam=1.0, kappa_i1=0.04, sigma_i0_sq=0.09)
        assert m[0] == pytest.approx(0.04, abs=1e-6)

    def test_stationary_start(self):
        t = np.linspace(0, 5, 100)
        m = bns_mean_variance(t, lam=2.0, kappa_i1=0.04, sigma_i0_sq=0.04)
        assert np.allclose(m, 0.04)

    def test_shape_preserved(self):
        t = np.linspace(0, 1, 50)
        m = bns_mean_variance(t, lam=0.5, kappa_i1=0.03, sigma_i0_sq=0.05)
        assert m.shape == t.shape


class TestBnsVarVariance:
    def test_at_zero(self):
        v = bns_var_variance(np.array([0.0]), lam=1.0, kappa_i2=0.1)
        assert v[0] == pytest.approx(0.0)

    def test_nonnegative(self):
        t = np.linspace(0, 5, 100)
        v = bns_var_variance(t, lam=1.0, kappa_i2=0.1)
        assert (v >= 0).all()

    def test_long_run_limit(self):
        v = bns_var_variance(np.array([1000.0]), lam=1.0, kappa_i2=0.2)
        assert v[0] == pytest.approx(0.1, rel=1e-5)  # kappa_i2/2


class TestBnsExpectedSigma:
    def test_approximately_sqrt_mean(self):
        """When kappa_i2=0, E[sigma] = sqrt(m)."""
        t = np.linspace(0.1, 2.0, 20)
        m = bns_mean_variance(t, lam=1.0, kappa_i1=0.04, sigma_i0_sq=0.04)
        E_sig = bns_expected_sigma(t, lam=1.0, kappa_i1=0.04, sigma_i0_sq=0.04, kappa_i2=0.0)
        assert np.allclose(E_sig, np.sqrt(m), rtol=1e-6)

    def test_nonnegative_small_correction(self):
        """The approximation stays non-negative when kappa_i2 is small and t is short."""
        t = np.linspace(0.01, 1.0, 50)
        E_sig = bns_expected_sigma(t, lam=1.0, kappa_i1=0.04, sigma_i0_sq=0.09, kappa_i2=0.001)
        assert (E_sig >= 0).all()


class TestBnsProductMeanIntegral:
    def test_stationary_case(self):
        """When sigma0_sq = kappa1, integral = prod(kappa1) * T."""
        T = 2.0
        kappa1 = np.array([0.04, 0.05, 0.03])
        sigma0_sq = kappa1.copy()
        result = bns_product_mean_integral(T, lam=1.0, kappa1=kappa1, sigma0_sq=sigma0_sq)
        expected = float(np.prod(kappa1)) * T
        assert result == pytest.approx(expected, rel=1e-8)

    def test_vs_numerical_integration(self):
        from scipy.integrate import quad
        kappa1 = np.array([0.04, 0.05, 0.03])
        sigma0_sq = np.array([0.08, 0.06, 0.05])
        lam = 0.5
        T = 1.5
        def integrand(t):
            m = np.exp(-lam * t) * (sigma0_sq - kappa1) + kappa1
            return m[0] * m[1] * m[2]
        numeric, _ = quad(integrand, 0, T)
        analytic = bns_product_mean_integral(T, lam, kappa1, sigma0_sq)
        assert analytic == pytest.approx(numeric, rel=1e-7)


class TestBnsExpectedGenVariance:
    def setup_method(self):
        self.lam = 0.5
        self.kappa1 = np.array([0.04, 0.05, 0.03])
        self.sigma0_sq = np.array([0.06, 0.07, 0.05])
        self.kappa_i2 = np.array([0.1, 0.1, 0.1])
        self.kappa02 = 0.5
        self.rho = np.ones(3) / np.sqrt(3)
        self.RB = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])

    def test_output_shape(self):
        T = np.linspace(0.01, 2.0, 30)
        grv = bns_expected_gen_variance(
            T, self.lam, self.kappa1, self.sigma0_sq,
            self.kappa_i2, self.kappa02, self.rho, self.RB
        )
        assert grv.shape == T.shape

    def test_zero_at_T_zero(self):
        grv = bns_expected_gen_variance(
            np.array([0.0]), self.lam, self.kappa1, self.sigma0_sq,
            self.kappa_i2, self.kappa02, self.rho, self.RB
        )
        assert grv[0] == 0.0

    def test_nonnegative_no_jumps(self):
        """With kappa02=0 (no jump correction), GRV = |R^B|/T * E0 is always non-negative."""
        T = np.linspace(0.1, 2.0, 20)
        grv = bns_expected_gen_variance(
            T, self.lam, self.kappa1, self.sigma0_sq,
            self.kappa_i2, kappa02=0.0, rho=self.rho, RB=self.RB
        )
        assert (grv >= 0).all()

    def test_stationary_case_analytical_check(self):
        """With sigma0_sq=kappa1, the E0 integral = prod(kappa1)*T, so result is analytical."""
        kappa1 = np.array([0.04, 0.05, 0.03])
        sigma0_sq = kappa1.copy()
        kappa_i2 = np.zeros(3)  # no variance-of-variance correction
        T = np.array([1.0])
        grv = bns_expected_gen_variance(
            T, self.lam, kappa1, sigma0_sq, kappa_i2, 0.0, self.rho, self.RB
        )
        RB_det = np.linalg.det(self.RB)
        expected = RB_det * float(np.prod(kappa1))
        assert grv[0] == pytest.approx(expected, rel=1e-5)


class TestBnsInference:
    def _make_calib_result(self):
        n = 100
        rng = np.random.default_rng(42)
        lam = 0.5
        kappa1 = np.array([0.04, 0.05, 0.03])
        sigma0_sq = np.array([0.06, 0.07, 0.05])
        kappa_i2 = np.array([0.1, 0.1, 0.1])
        kappa02 = 0.5
        rho = np.ones(3) / np.sqrt(3)
        RB = np.eye(3)
        T_array = np.linspace(0.01, 2.0, n)
        y_obs = bns_expected_gen_variance(T_array, lam, kappa1, sigma0_sq, kappa_i2, kappa02, rho, RB)
        det_series = pd.Series(y_obs, index=pd.date_range('2020', periods=n, freq='B'))
        return calibrate_bns(det_series, T_array, RB, n_starts=3, seed=0)

    def test_inference_keys(self):
        calib = self._make_calib_result()
        infer = bns_inference(calib)
        for key in ('params', 'std_errors', 't_stats', 't_values', 'p_values', 'param_names', 'dof', 'cov'):
            assert key in infer, f'Missing key: {key}'

    def test_p_values_in_range(self):
        calib = self._make_calib_result()
        infer = bns_inference(calib)
        pv = np.asarray(infer['p_values'])
        valid = np.isfinite(pv)
        assert ((pv[valid] >= 0.0) & (pv[valid] <= 1.0)).all()

    def test_t_values_alias(self):
        calib = self._make_calib_result()
        infer = bns_inference(calib)
        np.testing.assert_array_equal(infer['t_values'], infer['t_stats'])
