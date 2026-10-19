"""Tests for src/heston.py — Heston model functions."""
import numpy as np
import pytest
from src.heston import (
    heston_expected_variance_i,
    heston_expected_gen_variance,
    heston_residuals,
    heston_inference,
    calibrate_heston,
)


class TestHestonExpectedVarianceI:
    def test_at_zero(self):
        ev = heston_expected_variance_i(np.array([0.0]), 1.0, 0.04, 0.04)
        assert ev[0] == pytest.approx(0.04)

    def test_long_run_convergence(self):
        """As t -> inf, E[sigma^2] -> theta."""
        ev = heston_expected_variance_i(np.array([1000.0]), 1.0, 0.04, 0.09)
        assert ev[0] == pytest.approx(0.04, abs=1e-6)

    def test_stationary_start(self):
        """If sigma0_sq = theta, E[sigma^2] = theta for all t."""
        t = np.linspace(0, 5, 100)
        ev = heston_expected_variance_i(t, 2.0, 0.04, 0.04)
        assert np.allclose(ev, 0.04)

    def test_negative_kappa_raises(self):
        with pytest.raises(AssertionError):
            heston_expected_variance_i(np.array([1.0]), -0.1, 0.04, 0.04)


class TestHestonExpectedGenVariance:
    def setup_method(self):
        self.kappa = np.array([0.5, 0.7, 0.3])
        self.theta = np.array([0.04, 0.05, 0.03])
        self.sigma0 = self.theta.copy()
        self.corr_det = 0.5

    def test_output_shape(self):
        T = np.linspace(0.01, 2.0, 20)
        grv = heston_expected_gen_variance(T, self.kappa, self.theta, self.sigma0, self.corr_det)
        assert grv.shape == T.shape

    def test_nonnegative(self):
        T = np.linspace(0.01, 2.0, 50)
        grv = heston_expected_gen_variance(T, self.kappa, self.theta, self.sigma0, self.corr_det)
        assert (grv >= 0).all()

    def test_zero_at_T_zero(self):
        grv = heston_expected_gen_variance(np.array([0.0, 0.0]), self.kappa, self.theta, self.sigma0, self.corr_det)
        assert np.allclose(grv, 0.0)

    def test_stationary_constant_output(self):
        """With sigma0=theta, E[GRV] = |C| * prod(theta_i) for all T."""
        T = np.linspace(0.1, 5.0, 30)
        grv = heston_expected_gen_variance(T, self.kappa, self.theta, self.theta, self.corr_det)
        expected_val = self.corr_det * np.prod(self.theta)
        assert np.allclose(grv, expected_val, rtol=1e-6)

    def test_analytical_vs_numerical(self):
        """Compare analytic result to numerical integration (scipy quad)."""
        from scipy.integrate import quad
        T = 1.0
        grv_analytic = heston_expected_gen_variance(
            np.array([T]), self.kappa, self.theta, self.sigma0, self.corr_det
        )[0]
        def integrand(t):
            ev = self.theta + (self.sigma0 - self.theta) * np.exp(-self.kappa * t)
            return ev[0] * ev[1] * ev[2]
        integral, _ = quad(integrand, 0, T)
        grv_numeric = self.corr_det / T * integral
        assert grv_analytic == pytest.approx(grv_numeric, rel=1e-6)


class TestHestonResiduals:
    def test_zero_residuals_at_exact_params(self):
        """Residuals should be approximately zero if y_observed = model output."""
        kappa = np.array([0.5, 0.7, 0.3])
        theta = np.array([0.04, 0.05, 0.03])
        corr_det = 0.5
        T = np.linspace(0.1, 2.0, 30)
        params = np.concatenate([kappa, theta])
        y_model = heston_expected_gen_variance(T, kappa, theta, theta, corr_det)
        resid = heston_residuals(params, T, y_model, corr_det)
        assert np.allclose(resid, 0.0, atol=1e-10)


class TestHestonInference:
    def test_inference_keys(self):
        import pandas as pd
        rng = np.random.default_rng(99)
        T = np.linspace(0.01, 2.0, 100)
        corr_det = 0.5
        kappa = np.array([0.5, 0.7, 0.3])
        theta = np.array([0.04, 0.05, 0.03])
        y_true = heston_expected_gen_variance(T, kappa, theta, theta, corr_det) + rng.normal(0, 1e-6, 100)
        calib = calibrate_heston(
            pd.Series(y_true, index=pd.date_range('2020', periods=100, freq='B')),
            T, np.eye(3) * 0.5 + 0.5 * np.ones((3,3)) / 3,
            n_starts=3, seed=42
        )
        infer = heston_inference(calib)
        for key in ('param_names', 'params', 'std_errors', 't_values', 'p_values'):
            assert key in infer

    def test_p_values_in_range(self):
        import pandas as pd
        rng = np.random.default_rng(7)
        T = np.linspace(0.01, 2.0, 80)
        corr_det = 0.6
        kappa = np.array([0.5, 0.7, 0.3])
        theta = np.array([0.04, 0.05, 0.03])
        y_true = heston_expected_gen_variance(T, kappa, theta, theta, corr_det) + rng.normal(0, 1e-7, 80)
        C = np.eye(3)
        C[0, 1] = C[1, 0] = 0.3
        C[0, 2] = C[2, 0] = 0.2
        C[1, 2] = C[2, 1] = 0.1
        calib = calibrate_heston(
            pd.Series(y_true, index=pd.date_range('2020', periods=80, freq='B')),
            T, C, n_starts=2, seed=0
        )
        infer = heston_inference(calib)
        pv = np.asarray(infer['p_values'])
        valid = np.isfinite(pv)
        assert ((pv[valid] >= 0.0) & (pv[valid] <= 1.0)).all()
