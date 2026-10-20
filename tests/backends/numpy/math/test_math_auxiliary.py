"""Tests for numpy eager math extras ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numpy.eager.math_auxiliary import (
    _clamp,
    _eager_BroadcastedIota,
    _eager_Gradient,
    _eager_I0,
    _logspace,
    _np_constant_of_shape,
    _np_dtype_op,
    _np_frombuffer,
    _np_is_non_decreasing,
    _np_is_strictly_increasing,
    _np_l2_normalize,
    _np_rand,
    _np_reduce_euclidean_norm,
    _np_reduce_window,
    _np_test_eager_op,
    _np_unknown,
)
from ml_switcheroo_compiler.core.dtype import DType

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_constant_of_shape() -> None:
    """Test function test_constant_of_shape."""
    res = _np_constant_of_shape(np, (2, 2), 5.0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, typing.cast("EagerCallableProtocol", np.full)((2, 2), 5.0))


def test_reduce_window() -> None:
    """Test function test_reduce_window."""
    pass
    with pytest.raises(TypeError):
        _np_reduce_window(np, np.ones((2, 2)))


def test_test_eager_op() -> None:
    """Test function test_test_eager_op."""
    res = _np_test_eager_op(np)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, [1.0, 2.0, 3.0])


def test_unknown() -> None:
    """Test function test_unknown."""
    res = _np_unknown(np)
    assert res == 0.0


def test_rand() -> None:
    """Test function test_rand."""
    res = _np_rand(np, 2, 2)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res2 = _np_rand(np, 2, 2, dtype="float64")
    assert typing.cast("EagerTensorProtocol", getattr(res2, "dtype")) == np.float64
    res3 = _np_rand(np, 2, 2, dtype="bfloat16")
    assert typing.cast("EagerTensorProtocol", getattr(res3, "dtype")) == np.float32


def test_is_non_decreasing() -> None:
    """Test function test_is_non_decreasing."""
    assert _np_is_non_decreasing(np, np.array([1, 2, 2, 3]))
    assert not _np_is_non_decreasing(np, np.array([1, 2, 1]))
    assert _np_is_non_decreasing(np, np.array([1]))


def test_is_strictly_increasing() -> None:
    """Test function test_is_strictly_increasing."""
    assert _np_is_strictly_increasing(np, np.array([1, 2, 3]))
    assert not _np_is_strictly_increasing(np, np.array([1, 2, 2]))
    assert _np_is_strictly_increasing(np, np.array([1]))


def test_l2_normalize() -> None:
    """Test function test_l2_normalize."""
    x = np.array([3.0, 4.0])
    res = _np_l2_normalize(np, x)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [0.6, 0.8])


def test_reduce_euclidean_norm() -> None:
    """Test function test_reduce_euclidean_norm."""
    x = np.array([3.0, 4.0])
    res = _np_reduce_euclidean_norm(np, x)
    assert res == 5.0


def test_clamp() -> None:
    """Test function test_clamp."""
    x = np.array([1, 2, 3, 4, 5])
    res = _clamp(np, 2, x, 4)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [2, 2, 3, 4, 4])


def test_logspace() -> None:
    """Test function test_logspace."""
    res = _logspace(np, 1, 2, num=2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2

    class SpaceConfig:
        """Test class SpaceConfig."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.num = 2
            self.endpoint = True
            self.base = 10.0
            self.dtype = None
            self.axis = 0

    res2 = _logspace(np, 1, 2, typing.cast("EagerValue", SpaceConfig()))
    assert typing.cast("EagerCallableProtocol", len)(res2) == 2


def test_frombuffer() -> None:
    """Test function test_frombuffer."""
    b = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    res = _np_frombuffer(np, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1.0, 2.0])


def test_dtype_op() -> None:
    """Test function test_dtype_op."""
    res1 = _np_dtype_op(np, "float32")
    assert isinstance(res1, DType)
    res2 = _np_dtype_op(np, np.array([1.0, 2.0]))
    assert isinstance(res2, DType)
    res3 = _np_dtype_op(np, [1, 2])
    assert isinstance(res3, DType)


def test_eager_gradient() -> None:
    """Test function test_eager_gradient."""
    x = np.array([1, 2, 4, 7, 11])
    res = _eager_Gradient(np, x)
    expected = np.gradient(x)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_eager_i0() -> None:
    """Test function test_eager_i0."""
    x = np.array([0.0])
    res = _eager_I0(np, x)
    expected = np.i0(x)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_eager_broadcastediota() -> None:
    """Test function test_eager_broadcastediota."""
    res = _eager_BroadcastedIota(np, 3, (2, 3))
    expected = np.broadcast_to(np.arange(3), (2, 3))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_rand_int4() -> None:
    """Test function test_rand_int4."""
    res = _np_rand(np, 2, 2, dtype="int4")
    assert typing.cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int8


def test_np_rand_dt_str() -> None:
    """Test function test_np_rand_dt_str."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_auxiliary import _np_rand

    with pytest.raises(TypeError):
        res = _np_rand(np, 2, 2, dtype="fake_dtype")
    res = _np_rand(np, 2, 2, dtype="int4")
    assert res is not None
