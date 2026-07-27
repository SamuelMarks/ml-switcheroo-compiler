"""Tests for multivariate_normal."""

import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.continuous.multivariate_normal import MultivariateNormalOptions, multivariate_normal


def test_multivariate_normal_basic() -> None:
    """Test multivariate_normal function with default options and non-Tensor inputs."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.multivariate_normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = multivariate_normal("key", "mean", "cov")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("MultivariateNormal", ["key"], (), dtypes.DType.Float32, {"method": "cholesky"})


def test_multivariate_normal_with_tensors_and_options() -> None:
    """Test multivariate_normal function with Tensor inputs and custom options."""
    mock_mean = MagicMock(spec=Tensor)
    mock_cov = MagicMock(spec=Tensor)

    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.multivariate_normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        options = MultivariateNormalOptions(shape=(2, 2), dtype=dtypes.DType.Float64, method="svd")
        result = multivariate_normal("key", mock_mean, mock_cov, options=options)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("MultivariateNormal", ["key", mock_mean, mock_cov], (2, 2), dtypes.DType.Float64, {"method": "svd"})
