"""Tests for eager math_misc stubs."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import (
    _np_debuginfs,
    _np_debugnans,
    _np_rawmerge,
    _np_rawop,
    _np_rawswitch,
    _np_scanop,
    _np_switchop,
    _np_tensor,
    _np_tensorarrayread,
    _np_tensorarraystack,
    _np_tensorarraywrite,
    _np_vecdot,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_np_vecdot() -> None:
    """Test _np_vecdot."""
    x = np.array([1, 2])
    y = np.array([3, 4])
    assert _np_vecdot(ensure_eager_tensor(np), x, y) == 11

    # complex test
    x_c = np.array([1 + 1j])
    y_c = np.array([2 + 2j])
    assert _np_vecdot(ensure_eager_tensor(np), x_c, y_c) == (1 - 1j) * (2 + 2j)


def test_np_debuginfs() -> None:
    """Test _np_debuginfs."""
    x = np.array([1.0, 2.0])
    assert ensure_callable(np.array_equal)(ensure_callable(_np_debuginfs)(np, x), x)
    with pytest.raises(ValueError):
        _np_debuginfs(np, np.array([np.inf]))


def test_np_debugnans() -> None:
    """Test _np_debugnans."""
    x = np.array([1.0, 2.0])
    assert ensure_callable(np.array_equal)(ensure_callable(_np_debugnans)(np, x), x)
    with pytest.raises(ValueError):
        _np_debugnans(np, np.array([np.nan]))


def test_np_tensor() -> None:
    """Test _np_tensor."""
    assert ensure_callable(np.array_equal)(ensure_callable(_np_tensor)(np, [1, 2]), ensure_callable(np.array)([1, 2]))
    assert ensure_callable(np.array_equal)(ensure_callable(_np_tensor)(np), ensure_callable(np.array)([]))


def test_np_tensorarray() -> None:
    """Test _np_tensorarray operations."""
    arr = _np_tensorarraywrite(ensure_eager_tensor(np), [], 0, 42)
    assert arr == [42]
    arr = _np_tensorarraywrite(ensure_eager_tensor(np), arr, 2, 100)
    assert arr == [42, None, 100]

    assert _np_tensorarrayread(ensure_eager_tensor(np), arr, 0) == 42
    assert _np_tensorarrayread(ensure_eager_tensor(np), arr, 2) == 100

    stacked = _np_tensorarraystack(ensure_eager_tensor(np), [np.array([1]), np.array([2])])
    assert ensure_callable(np.array_equal)(stacked, ensure_callable(np.array)([[1], [2]]))


def test_raw_and_switch_ops() -> None:
    # rawmerge
    """Test function test_raw_and_switch_ops."""
    res1, i1 = ensure_callable(_np_rawmerge)(np, [ensure_callable(np.array)([1]), ensure_callable(np.array)([2])])
    assert ensure_callable(np.array_equal)(res1, ensure_callable(np.array)([1]))
    assert ensure_eager_tensor(i1) == 0

    res2, i2 = ensure_callable(_np_rawmerge)(np, ensure_callable(np.array)([1]))
    assert ensure_callable(np.array_equal)(res2, ensure_callable(np.array)([1]))

    res3, i3 = ensure_callable(_np_rawmerge)(np)
    assert ensure_eager_tensor(res3) is None
    assert i3 == -1

    # rawop
    assert _np_rawop(ensure_eager_tensor(np), 42) == 42
    assert _np_rawop(ensure_eager_tensor(np)) is None

    # rawswitch
    rs_pair1 = ensure_tensor_pair(ensure_callable(_np_rawswitch)(np, 42, False))
    assert rs_pair1[0] == 42
    assert ensure_eager_tensor(rs_pair1)[1] is None

    rs_pair2 = ensure_tensor_pair(ensure_callable(_np_rawswitch)(np, 42, pred=True))
    assert ensure_eager_tensor(rs_pair2)[0] is None
    assert rs_pair2[1] == 42

    # switchop
    out_f, out_t = ensure_tensor_pair(ensure_callable(_np_switchop)(np, 42, False))
    assert out_f == 42

    out_f, out_t = ensure_tensor_pair(ensure_callable(_np_switchop)(np, 42, pred=True))
    assert out_t == 42


def test_scan_op() -> None:
    """Test function test_scan_op."""

    def fn(a: EagerValue, b: EagerValue) -> EagerValue:
        """Add two eager values.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            EagerValue: Computed output value from `fn`.
        """
        return ensure_eager_tensor(a) + ensure_eager_tensor(b)

    elems = np.array([1, 2, 3])

    # With initial acc
    res = _np_scanop(ensure_eager_tensor(np), fn, elems, 10)
    assert ensure_callable(np.array_equal)(res, ensure_callable(np.array)([11, 13, 16]))

    # Without initial acc
    res2 = _np_scanop(ensure_eager_tensor(np), fn, elems)
    assert ensure_callable(np.array_equal)(res2, ensure_callable(np.array)([1, 3, 6]))

    # None args
    assert _np_scanop(ensure_eager_tensor(np)) is None
    assert _np_scanop(ensure_eager_tensor(np), fn) == fn
