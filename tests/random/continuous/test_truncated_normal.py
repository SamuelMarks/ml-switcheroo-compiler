"""Tests for truncated_normal."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.truncated_normal import truncated_normal


def test_truncated_normal() -> None:
    """Test truncated_normal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.truncated_normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = truncated_normal("key", "lower", "upper", shape=(2, 2), dtype=dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomTruncatedNormal", ["key"], (2, 2), dtypes.DType.Float64, {"lower": "lower", "upper": "upper"})


def test_truncated_normal_default_shape_and_dtype() -> None:
    """Test truncated_normal function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.truncated_normal"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = truncated_normal("key", "lower", "upper")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomTruncatedNormal", ["key"], (), dtypes.DType.Float32, {"lower": "lower", "upper": "upper"})
