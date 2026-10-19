"""Tests for src/metrics.py — error metric functions."""
import numpy as np
import pytest
from src.metrics import (
    compute_rmse, compute_ape, compute_aae, compute_arpe,
    compute_r_squared, compute_all_metrics, percent_improvement,
)


class TestComputeRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        assert compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, 50)
        y_pred = rng.uniform(0, 1, 50)
        assert compute_rmse(y_true, y_pred) >= 0.0

    def test_scalar_inputs(self):
        assert compute_rmse(np.array([3.0]), np.array([0.0])) == pytest.approx(3.0)


class TestComputeAPE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_ape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])  # 100% error each
        assert compute_ape(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_true_skipped(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([5.0, 1.0, 2.0])
        assert compute_ape(y_true, y_pred) == pytest.approx(0.0)

    def test_all_zero_returns_nan(self):
        assert np.isnan(compute_ape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))

    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0.01, 1.0, 50)
        y_pred = rng.uniform(0.01, 1.0, 50)
        assert compute_ape(y_true, y_pred) >= 0.0


class TestComputeAAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_aae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert compute_aae(y_true, y_pred) == pytest.approx(2.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(2)
        y_true = rng.uniform(0, 1, 50)
        y_pred = rng.uniform(0, 1, 50)
        assert compute_aae(y_true, y_pred) >= 0.0


class TestComputeRSquared:
    def test_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction_is_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, np.mean(y_true))
        assert compute_r_squared(y_true, y_pred) == pytest.approx(0.0)

    def test_constant_true_returns_nan(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        assert np.isnan(compute_r_squared(y_true, y_pred))


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_all_metrics(y, y)
        for key in ('rmse', 'ape', 'aae', 'arpe', 'r_squared', 'label'):
            assert key in m

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_all_metrics(y, y, label='test')
        assert m['rmse'] == pytest.approx(0.0)
        assert m['ape'] == pytest.approx(0.0)
        assert m['aae'] == pytest.approx(0.0)
        assert m['label'] == 'test'


class TestPercentImprovement:
    def test_positive_improvement(self):
        assert percent_improvement(10.0, 6.0) == pytest.approx(40.0)

    def test_no_improvement(self):
        assert percent_improvement(5.0, 5.0) == pytest.approx(0.0)

    def test_zero_heston_returns_nan(self):
        assert np.isnan(percent_improvement(0.0, 1.0))

    def test_negative_improvement(self):
        assert percent_improvement(5.0, 10.0) == pytest.approx(-100.0)
