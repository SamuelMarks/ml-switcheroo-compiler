"""Tests for normal."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.normal import normal


def test_normal() -> None:
    """Test normal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = normal("key", shape=(2, 2), dtype=dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomNormal", ["key"], (2, 2), dtypes.DType.Float64)


def test_normal_default_shape_and_dtype() -> None:
    """Test normal function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = normal("key")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomNormal", ["key"], (), dtypes.DType.Float32)
