"""Tests for eager math_misc stubs."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.math_misc import (
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


def test_np_vecdot() -> None:
    """Test _np_vecdot."""
    x = np.array([1, 2])
    y = np.array([3, 4])
    assert _np_vecdot(np, x, y) == 11

    # complex test
    x_c = np.array([1 + 1j])
    y_c = np.array([2 + 2j])
    assert _np_vecdot(np, x_c, y_c) == (1 - 1j) * (2 + 2j)


def test_np_debuginfs() -> None:
    """Test _np_debuginfs."""
    x = np.array([1.0, 2.0])
    assert np.array_equal(_np_debuginfs(np, x), x)
    with pytest.raises(ValueError):
        _np_debuginfs(np, np.array([np.inf]))


def test_np_debugnans() -> None:
    """Test _np_debugnans."""
    x = np.array([1.0, 2.0])
    assert np.array_equal(_np_debugnans(np, x), x)
    with pytest.raises(ValueError):
        _np_debugnans(np, np.array([np.nan]))


def test_np_tensor() -> None:
    """Test _np_tensor."""
    assert np.array_equal(_np_tensor(np, [1, 2]), np.array([1, 2]))
    assert np.array_equal(_np_tensor(np), np.array([]))


def test_np_tensorarray() -> None:
    """Test _np_tensorarray operations."""
    arr = _np_tensorarraywrite(np, [], 0, 42)
    assert arr == [42]
    arr = _np_tensorarraywrite(np, arr, 2, 100)
    assert arr == [42, None, 100]

    assert _np_tensorarrayread(np, arr, 0) == 42
    assert _np_tensorarrayread(np, arr, 2) == 100

    stacked = _np_tensorarraystack(np, [np.array([1]), np.array([2])])
    assert np.array_equal(stacked, np.array([[1], [2]]))


def test_raw_and_switch_ops():
    # rawmerge
    res1, i1 = _np_rawmerge(np, [np.array([1]), np.array([2])])
    assert np.array_equal(res1, np.array([1]))
    assert i1 == 0

    res2, i2 = _np_rawmerge(np, np.array([1]))
    assert np.array_equal(res2, np.array([1]))

    res3, i3 = _np_rawmerge(np)
    assert res3 is None
    assert i3 == -1

    # rawop
    assert _np_rawop(np, 42) == 42
    assert _np_rawop(np) is None

    # rawswitch
    out_f, out_t = _np_rawswitch(np, 42, False)
    assert out_f == 42
    assert out_t is None

    out_f, out_t = _np_rawswitch(np, 42, pred=True)
    assert out_f is None
    assert out_t == 42

    # switchop
    out_f, out_t = _np_switchop(np, 42, False)
    assert out_f == 42

    out_f, out_t = _np_switchop(np, 42, pred=True)
    assert out_t == 42


def test_scan_op():
    fn = lambda a, b: a + b
    elems = np.array([1, 2, 3])

    # With initial acc
    res = _np_scanop(np, fn, elems, 10)
    assert np.array_equal(res, np.array([11, 13, 16]))

    # Without initial acc
    res2 = _np_scanop(np, fn, elems)
    assert np.array_equal(res2, np.array([1, 3, 6]))

    # None args
    assert _np_scanop(np) is None
    assert _np_scanop(np, fn) == fn
