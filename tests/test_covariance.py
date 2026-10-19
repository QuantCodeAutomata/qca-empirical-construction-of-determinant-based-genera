"""Tests for src/covariance.py — covariance matrix estimation."""
import numpy as np
import pandas as pd
import pytest
from src.covariance import (
    rolling_covariance_determinants,
    nonoverlapping_covariance_determinants,
    covariance_matrix_diagnostics,
    compute_horizon_map,
    estimate_windowed_covariances,
)


def _make_returns(n: int = 50, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    data = rng.standard_normal((n, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f'A{i}' for i in range(n_assets)])


class TestRollingCovarianceDeterminants:
    def test_length(self):
        ret = _make_returns(50, 3)
        det = rolling_covariance_determinants(ret, window=10)
        assert len(det) == 50 - 10 + 1

    def test_dtype_float64(self):
        det = rolling_covariance_determinants(_make_returns(30, 3), window=5)
        assert det.dtype == np.float64

    def test_nonnegative(self):
        """Determinants of sample covariance matrices must be >= -tol for PSD."""
        det = rolling_covariance_determinants(_make_returns(40, 3), window=10)
        assert (det.values >= -1e-12).all()

    def test_series_name(self):
        det = rolling_covariance_determinants(_make_returns(30, 3), window=5)
        assert det.name == 'det_cov'

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            rolling_covariance_determinants(_make_returns(30, 3), window=1)

    def test_insufficient_rows_raises(self):
        with pytest.raises(ValueError):
            rolling_covariance_determinants(_make_returns(5, 3), window=10)

    def test_single_window(self):
        ret = _make_returns(10, 3)
        det = rolling_covariance_determinants(ret, window=10)
        assert len(det) == 1


class TestNonoverlapCovarianceDeterminants:
    def test_length(self):
        ret = _make_returns(50, 3)
        det = nonoverlapping_covariance_determinants(ret, window=10)
        assert len(det) == 5

    def test_incomplete_block_discarded(self):
        ret = _make_returns(53, 3)
        det = nonoverlapping_covariance_determinants(ret, window=10)
        assert len(det) == 5  # 53 // 10 = 5

    def test_nonnegative(self):
        det = nonoverlapping_covariance_determinants(_make_returns(50, 3), window=10)
        assert (det.values >= -1e-12).all()


class TestComputeHorizonMap:
    def test_shape_matches_series(self):
        det = rolling_covariance_determinants(_make_returns(40, 3), window=10)
        T = compute_horizon_map(det)
        assert len(T) == len(det)

    def test_values_positive(self):
        det = rolling_covariance_determinants(_make_returns(40, 3), window=10)
        T = compute_horizon_map(det)
        assert (T > 0).all()

    def test_starts_at_1_over_252(self):
        det = rolling_covariance_determinants(_make_returns(40, 3), window=10)
        T = compute_horizon_map(det)
        assert T[0] == pytest.approx(1.0 / 252.0)

    def test_invalid_freq_raises(self):
        det = rolling_covariance_determinants(_make_returns(30, 3), window=5)
        with pytest.raises(ValueError):
            compute_horizon_map(det, freq='monthly')


class TestCovarianceMatrixDiagnostics:
    def test_returns_dataframe(self):
        ret = _make_returns(40, 3)
        df = covariance_matrix_diagnostics(ret, window=10, variant='rolling')
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        ret = _make_returns(40, 3)
        df = covariance_matrix_diagnostics(ret, window=10, variant='rolling')
        for col in ('det', 'min_eigenvalue', 'condition_number', 'is_singular', 'log_det'):
            assert col in df.columns

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            covariance_matrix_diagnostics(_make_returns(30, 3), window=5, variant='bad')
