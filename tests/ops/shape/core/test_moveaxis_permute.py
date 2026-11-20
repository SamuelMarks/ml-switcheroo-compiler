"""Tests for shape ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import moveaxis, permute
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import coerce_shape, ensure_tensor, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _t(data: BackendArg, shape: tuple[BackendArg, ...]) -> Tensor:
    """Helper to create test tensor.

    Args:
        data: Test parameter `data`.
        shape: Test parameter `shape`.

    Returns:
        Tensor: Computed output value from `_t`.
    """
    return ensure_tensor(Tensor(data, TensorConfig(coerce_shape(shape), "float32", "cpu")))


def test_permute_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2, 3]]), (1, 3))
    out = permute(x, (1, 0))
    assert out.shape == (3, 1)


def test_permute_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2, 3]]), (1, 3))
        out = permute(x, (1, 0))
        assert out.shape == (3, 1)
        out2 = permute(x, None)
        assert out2.shape == (3, 1)
    finally:
        global_tracing_state.stop_tracing()


def test_moveaxis_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2, 3]]), (1, 3))
    out = moveaxis(x, 1, 0)
    assert extract_test_shape(out) == (3, 1)


def test_moveaxis_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2, 3]]), (1, 3))
        out = moveaxis(x, 1, 0)
        assert extract_test_shape(out) == (3, 1)
        out2 = moveaxis(x, [1], [0])
        assert extract_test_shape(out2) == (3, 1)
        out3 = moveaxis(x, -1, -2)
        assert extract_test_shape(out3) == (3, 1)
    finally:
        global_tracing_state.stop_tracing()
