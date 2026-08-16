"""Tests for uniform."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.uniform import uniform


def test_uniform() -> None:
    """Test uniform function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.uniform"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = uniform("key", shape=(2, 2), dtype=dtypes.DType.Float64, minval=-1.0, maxval=2.0)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomUniform", ["key"], (2, 2), dtypes.DType.Float64, {"minval": -1.0, "maxval": 2.0})


def test_uniform_default_shape_and_dtype() -> None:
    """Test uniform function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.uniform"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = uniform("key")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomUniform", ["key"], (), dtypes.DType.Float32, {"minval": 0.0, "maxval": 1.0})
