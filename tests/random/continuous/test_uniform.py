"""Tests for uniform."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.continuous.uniform import uniform

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_uniform() -> None:
    """Test uniform function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.uniform"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = uniform("key", shape=(2, 2), dtype=dtypes.DType.Float64, minval=-1.0, maxval=2.0)
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_emit.assert_called_once_with("RandomUniform", ["key"], (2, 2), dtypes.DType.Float64, {"minval": -1.0, "maxval": 2.0})


def test_uniform_default_shape_and_dtype() -> None:
    """Test uniform function with default shape and dtype."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.uniform"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = uniform("key")
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_emit.assert_called_once_with("RandomUniform", ["key"], (), dtypes.DType.Float32, {"minval": 0.0, "maxval": 1.0})
