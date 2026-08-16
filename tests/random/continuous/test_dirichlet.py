"""Tests for dirichlet."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.dirichlet import dirichlet


def test_dirichlet() -> None:
    """Test dirichlet function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.dirichlet"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = dirichlet("key", "alpha", shape=(2, 2), dtype=dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Dirichlet", ["key", "alpha"], (2, 2), dtypes.DType.Float64)


def test_dirichlet_default_shape_and_dtype() -> None:
    """Test dirichlet function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.dirichlet"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = dirichlet("key", "alpha")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Dirichlet", ["key", "alpha"], (), dtypes.DType.Float32)
