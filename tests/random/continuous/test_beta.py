"""Tests for beta."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.beta import beta


def test_beta() -> None:
    """Test beta function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.beta"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = beta("key", "a", "b", shape=(2, 2), dtype=dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Beta", ["key", "a", "b"], (2, 2), dtypes.DType.Float64)


def test_beta_default_shape_and_dtype() -> None:
    """Test beta function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.beta"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = beta("key", "a", "b")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Beta", ["key", "a", "b"], (), dtypes.DType.Float32)
