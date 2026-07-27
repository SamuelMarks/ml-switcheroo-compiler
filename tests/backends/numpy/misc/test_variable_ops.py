"""Tests for numpy eager variable ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import (
    _np_assign,
    _np_assign_variable,
    _np_bitcast,
    _np_cast,
    _np_read_variable,
)


def test_assign() -> None:
    res = _np_assign(np, 1, 2)
    assert res == 2


def test_cast() -> None:
    x = np.array([1.5, 2.5])
    res = _np_cast(np, x, "int32")
    assert res.dtype == np.int32
    np.testing.assert_allclose(res, [1, 2])

    class MockDtype:
        def __init__(self, val):
            self.value = val

    res2 = _np_cast(np, x, MockDtype("bfloat16"))
    assert res2.dtype == np.float32

    res3 = _np_cast(np, x, MockDtype("int4"))
    assert res3.dtype == np.int8


def test_bitcast() -> None:
    x = np.array([1.0], dtype=np.float32)
    res = _np_bitcast(np, x, "int32")
    assert res.dtype == np.int32

    class MockDtype:
        def __init__(self, val):
            self.value = val

    res2 = _np_bitcast(np, x, MockDtype("int32"))
    assert res2.dtype == np.int32


def test_read_variable() -> None:
    res = _np_read_variable(np, 123)
    assert res == 123


def test_assign_variable() -> None:
    res = _np_assign_variable(np, "ref", 456)
    assert res == 456
