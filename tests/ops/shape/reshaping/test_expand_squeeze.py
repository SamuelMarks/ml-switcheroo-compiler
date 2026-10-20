"""Tests for shape ops expand_dims and squeeze."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import atleast_1d, atleast_2d, atleast_3d, expand_dims, squeeze
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _t(data: EagerValue, shape: tuple[EagerValue, ...]) -> Tensor:
    """Helper to create test tensor.

    Args:
        data: Test parameter `data`.
        shape: Test parameter `shape`.
    """
    return typing.cast("Tensor", typing.cast("EagerCallableProtocol", Tensor)(data, typing.cast("EagerCallableProtocol", TensorConfig)(shape, "float32", "cpu")))


def test_expand_dims_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    out = expand_dims(x, axis=0)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 3)


def test_expand_dims_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([1, 2, 3]), (3,))
        out = expand_dims(x, axis=1)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (3, 1)
    finally:
        global_tracing_state.stop_tracing()


def test_expand_dims_no_shape() -> None:
    """Test no shape."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        out = expand_dims(5, axis=0)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == ("None",)
    finally:
        global_tracing_state.stop_tracing()


def test_squeeze_eager() -> None:
    """Test eager."""
    config.eager_mode = True
    x = _t(np.array([[1, 2, 3]]), (1, 3))
    out = squeeze(x, axis=0)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (3,)


def test_squeeze_tracing() -> None:
    """Test tracing."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[1, 2, 3]]), (1, 3))
        out = squeeze(x, axis=0)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (3,)
    finally:
        global_tracing_state.stop_tracing()


def test_squeeze_no_shape() -> None:
    """Test no shape."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        out = squeeze(5, axis=0)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == ()
    finally:
        global_tracing_state.stop_tracing()


def test_squeeze_all() -> None:
    """Test all axes."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array([[[1]]]), (1, 1, 1))
        out = squeeze(x)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == ()
    finally:
        global_tracing_state.stop_tracing()


def test_squeeze_eager_negative_axis() -> None:
    """Test eager negative."""
    config.eager_mode = True
    x = _t(np.array([[1, 2, 3]]), (1, 3))
    out = squeeze(x, axis=-2)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (3,)


def test_expand_dims_eager_negative_axis() -> None:
    """Test eager negative."""
    config.eager_mode = True
    x = _t(np.array([1, 2, 3]), (3,))
    out = expand_dims(x, axis=-1)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (3, 1)


def test_atleast_1d_eager() -> None:
    """Test eager atleast_1d."""
    config.eager_mode = True
    x = _t(np.array(5), ())
    out = atleast_1d(x)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1,)


def test_atleast_1d_tracing() -> None:
    """Test tracing atleast_1d."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array(5), ())
        out = atleast_1d(x)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1,)

        x2 = _t(np.array([1]), (1,))
        out2 = atleast_1d(x2)
        assert typing.cast("EagerTensorProtocol", getattr(out2, "shape")) == (1,)
    finally:
        global_tracing_state.stop_tracing()


def test_atleast_2d_eager() -> None:
    """Test eager atleast_2d."""
    config.eager_mode = True
    x = _t(np.array([5]), (1,))
    out = atleast_2d(x)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 1)


def test_atleast_2d_tracing() -> None:
    """Test tracing atleast_2d."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array(5), ())
        out = atleast_2d(x)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 1)

        x2 = _t(np.array([5]), (1,))
        out2 = atleast_2d(x2)
        assert typing.cast("EagerTensorProtocol", getattr(out2, "shape")) == (1, 1)

        x3 = _t(np.array([[5]]), (1, 1))
        out3 = atleast_2d(x3)
        assert typing.cast("EagerTensorProtocol", getattr(out3, "shape")) == (1, 1)
    finally:
        global_tracing_state.stop_tracing()


def test_atleast_3d_eager() -> None:
    """Test eager atleast_3d."""
    config.eager_mode = True
    x = _t(np.array([[5]]), (1, 1))
    out = atleast_3d(x)
    assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 1, 1)


def test_atleast_3d_tracing() -> None:
    """Test tracing atleast_3d."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        x = _t(np.array(5), ())
        out = atleast_3d(x)
        assert typing.cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 1, 1)

        x1 = _t(np.array([5]), (1,))
        out1 = atleast_3d(x1)
        assert typing.cast("EagerTensorProtocol", getattr(out1, "shape")) == (1, 1, 1)

        x2 = _t(np.array([[5]]), (1, 1))
        out2 = atleast_3d(x2)
        assert typing.cast("EagerTensorProtocol", getattr(out2, "shape")) == (1, 1, 1)

        x3 = _t(np.array([[[5]]]), (1, 1, 1))
        out3 = atleast_3d(x3)
        assert typing.cast("EagerTensorProtocol", getattr(out3, "shape")) == (1, 1, 1)
    finally:
        global_tracing_state.stop_tracing()
