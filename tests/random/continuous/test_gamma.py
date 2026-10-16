"""Tests for gamma."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.gamma import gamma

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_gamma() -> None:
    """Test gamma function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.gamma"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = cast("Callable[..., EagerTensorProtocol]", gamma)("key", "a", shape=(2, 2), dtype=dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Gamma", ["key", "a"], (2, 2), dtypes.DType.Float64)


def test_gamma_default_dtype() -> None:
    """Test gamma function with default dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.gamma"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = cast("Callable[..., EagerTensorProtocol]", gamma)("key", "a", shape=(2, 2))
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("Gamma", ["key", "a"], (2, 2), dtypes.DType.Float32)
