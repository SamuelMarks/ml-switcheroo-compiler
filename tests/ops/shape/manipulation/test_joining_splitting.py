"""Tests for shape ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import append, column_stack, dsplit, dstack, hsplit, hstack, vsplit, vstack
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import coerce_operand_list, coerce_shape, ensure_tensor, extract_test_shape

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


def test_append_eager() -> None:
    """Test append_eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    y = _t(np.array([4, 5]), (2,))
    out = append(x, y)
    assert extract_test_shape(out) == (5,)


def test_append_tracing() -> None:
    """Test append_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2, 3]), (3,))
        y = _t(np.array([4, 5]), (2,))
        out = append(x, y)
        assert extract_test_shape(out) == (5,)
    finally:
        global_tracing_state.stop_tracing()


def test_column_stack_eager() -> None:
    """Test column_stack_eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    y = _t(np.array([4, 5, 6]), (3,))
    out = column_stack([x, y])
    assert extract_test_shape(out) == (3, 2)


def test_column_stack_tracing() -> None:
    """Test column_stack_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2, 3]), (3,))
        y = _t(np.array([4, 5, 6]), (3,))
        column_stack([x, y])
    finally:
        global_tracing_state.stop_tracing()
    assert callable(global_tracing_state.start_tracing)


def test_dstack_eager() -> None:
    """Test dstack_eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
    y = _t(np.array([[5, 6], [7, 8]]), (2, 2))
    out = dstack([x, y])
    assert extract_test_shape(out) == (2, 2, 2)


def test_dstack_tracing() -> None:
    """Test dstack_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2], [3, 4]]), (2, 2))
        y = _t(np.array([[5, 6], [7, 8]]), (2, 2))
        dstack([x, y])
    finally:
        global_tracing_state.stop_tracing()
    assert callable(global_tracing_state.start_tracing)


def test_hstack_eager() -> None:
    """Test hstack_eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    y = _t(np.array([3, 4]), (2,))
    out = hstack([x, y])
    assert extract_test_shape(out) == (4,)


def test_hstack_tracing() -> None:
    """Test hstack_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        y = _t(np.array([3, 4]), (2,))
        hstack([x, y])
    finally:
        global_tracing_state.stop_tracing()
    assert callable(global_tracing_state.start_tracing)


def test_vstack_eager() -> None:
    """Test vstack_eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2]), (2,))
    y = _t(np.array([3, 4]), (2,))
    out = vstack([x, y])
    assert extract_test_shape(out) == (2, 2)


def test_vstack_tracing() -> None:
    """Test vstack_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2]), (2,))
        y = _t(np.array([3, 4]), (2,))
        vstack([x, y])
    finally:
        global_tracing_state.stop_tracing()
    assert callable(global_tracing_state.start_tracing)


def test_dsplit_eager() -> None:
    """Test dsplit_eager."""
    config.eager_mode = True
    x = _t(np.ones((2, 2, 2)), (2, 2, 2))
    out = dsplit(x, 2)
    assert len(coerce_operand_list(out)) == 2
    assert extract_test_shape(coerce_operand_list(out)[0]) == (2, 2, 1)


def test_dsplit_tracing() -> None:
    """Test dsplit_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.ones((2, 2, 2)), (2, 2, 2))
        out = dsplit(x, 2)
        assert len(coerce_operand_list(out)) == 2
        assert extract_test_shape(coerce_operand_list(out)[0]) == (2, 2, 1)
    finally:
        global_tracing_state.stop_tracing()


def test_hsplit_eager() -> None:
    """Test hsplit_eager."""
    config.eager_mode = True
    x = _t(np.ones((2, 2)), (2, 2))
    out = hsplit(x, 2)
    assert len(coerce_operand_list(out)) == 2
    assert extract_test_shape(coerce_operand_list(out)[0]) == (2, 1)


def test_hsplit_tracing() -> None:
    """Test hsplit_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.ones((2, 2)), (2, 2))
        out = hsplit(x, 2)
        assert len(coerce_operand_list(out)) == 2
        assert extract_test_shape(coerce_operand_list(out)[0]) == (2, 1)
    finally:
        global_tracing_state.stop_tracing()


def test_vsplit_eager() -> None:
    """Test vsplit_eager."""
    config.eager_mode = True
    x = _t(np.ones((2, 2)), (2, 2))
    out = vsplit(x, 2)
    assert len(coerce_operand_list(out)) == 2
    assert extract_test_shape(coerce_operand_list(out)[0]) == (1, 2)


def test_vsplit_tracing() -> None:
    """Test vsplit_tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.ones((2, 2)), (2, 2))
        out = vsplit(x, 2)
        assert len(coerce_operand_list(out)) == 2
        assert extract_test_shape(coerce_operand_list(out)[0]) == (1, 2)
    finally:
        global_tracing_state.stop_tracing()
