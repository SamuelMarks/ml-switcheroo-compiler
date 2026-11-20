"""Behavioral unit tests for math nan."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_nan as nan_module
from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import (
    _np_divide_no_nan,
    _np_multiply_no_nan,
    _np_nanmean,
    _np_nanmedian,
    _np_reciprocal_no_nan,
    _np_squared_difference,
    _np_xdivy,
    _np_zero_fraction,
    _xlogy,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_xlog1py
from tests.testing_utils import MockTestStub, NumpyTestArray, coerce_operand_list, require_callable


def test_divide_no_nan() -> None:
    """Test divide_no_nan."""
    x = np.array([1.0, 1.0])
    y = np.array([1.0, 0.0])
    res = _np_divide_no_nan(np, x, y)
    np.testing.assert_allclose(ensure_ndarray(res), [1.0, 0.0])


def test_multiply_no_nan() -> None:
    """Test multiply_no_nan."""
    x = np.array([1.0, 1.0])
    y = np.array([1.0, 0.0])
    res = _np_multiply_no_nan(np, x, y)
    np.testing.assert_allclose(ensure_ndarray(res), [1.0, 0.0])


def test_squared_difference() -> None:
    """Test squared_difference."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 0.0])
    res = _np_squared_difference(np, x, y)
    np.testing.assert_allclose(ensure_ndarray(res), [0.0, 4.0])


def test_xdivy() -> None:
    """Test xdivy."""
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 2.0])
    res = _np_xdivy(np, x, y)
    np.testing.assert_allclose(ensure_ndarray(res), [0.0, 1.0])


def test_xlog1py() -> None:
    """Test xlog1py."""
    x = np.array([0.0, 1.0])
    y = np.array([1.0, np.e - 1])
    res = _np_xlog1py(np, x, y)
    np.testing.assert_allclose(res, [0.0, 1.0])


def test_reciprocal_no_nan() -> None:
    """Test reciprocal_no_nan."""
    x = np.array([0.0, 2.0])
    res = _np_reciprocal_no_nan(np, x)
    np.testing.assert_allclose(ensure_ndarray(res), [0.0, 0.5])


def test_zero_fraction() -> None:
    """Test zero_fraction."""
    x = np.array([0.0, 1.0, 2.0])
    res = _np_zero_fraction(np, x)
    assert res == 1.0 / 3.0

    # test empty array
    x_empty = np.array([])
    res_empty = _np_zero_fraction(np, x_empty)
    assert np.isnan(ensure_ndarray(res_empty))


def test_xlogy() -> None:
    """Test xlogy."""
    x = np.array([0.0, 1.0])
    y = np.array([1.0, np.e])
    res = _xlogy(x, y)
    np.testing.assert_allclose(res, [0.0, 1.0])


def test_nanmean() -> None:
    """Test nanmean."""
    a = np.array([1.0, np.nan, 3.0])
    res = _np_nanmean(np, a)
    assert res == 2.0


def test_nanmedian() -> None:
    """Test nanmedian."""
    a = np.array([1.0, np.nan, 3.0])
    res = _np_nanmedian(np, a)
    assert res == 2.0


def test_math_nan_ops() -> None:
    """Test NaN-safe operations directly and through eager registry.

    Returns:
        None
    """
    x = np.array([4.0, 2.0, 0.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 0.0], dtype=np.float32)

    assert np.allclose(ensure_ndarray(nan_module._np_divide_no_nan(np, x, y)), [2.0, 0.5, 0.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("DivideNoNan")(np, x, y), [2.0, 0.5, 0.0])

    assert np.allclose(ensure_ndarray(nan_module._np_multiply_no_nan(np, x, y)), [8.0, 8.0, 0.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("MultiplyNoNan")(np, x, y), [8.0, 8.0, 0.0])

    assert np.allclose(ensure_ndarray(nan_module._np_squared_difference(np, x, y)), (x - y) ** 2)
    assert np.allclose(require_callable(numpy_eager_registry.get)("SquaredDifference")(np, x, y), (x - y) ** 2)

    assert np.allclose(ensure_ndarray(nan_module._np_xdivy(np, x, y)), [2.0, 0.5, 0.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Xdivy")(np, x, y), [2.0, 0.5, 0.0])

    expected_xlog1py = np.array([4.0 * np.log1p(2.0), 2.0 * np.log1p(4.0), 0.0])
    assert np.allclose(ensure_ndarray(nan_module._np_xlog1py(np, x, y)), expected_xlog1py)
    assert np.allclose(require_callable(numpy_eager_registry.get)("Xlog1py")(np, x, y), expected_xlog1py)

    assert np.allclose(ensure_ndarray(nan_module._np_reciprocal_no_nan(np, x)), [0.25, 0.5, 0.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("ReciprocalNoNan")(np, x), [0.25, 0.5, 0.0])

    # ZeroFraction - non-empty and empty
    class _ZeroFracStub(MockTestStub):
        """Stub backend module for `ZeroFraction`."""

        @staticmethod
        def sum(a: BackendArg) -> int:
            """Return mock zero count.

            Args:
                a: Equality mask operand.

            Returns:
                int: Mock zero count `2`.
            """
            del a
            return 2

        @staticmethod
        def equal(a: BackendArg, b: BackendArg) -> list[bool]:
            """Return mock equality mask.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                list[bool]: Mock boolean mask.
            """
            del a, b
            return [True, False, True, False]

        @staticmethod
        def size(a: BackendArg) -> int:
            """Return mock element count.

            Args:
                a: Input sequence or array.

            Returns:
                int: `4` if non-empty, else `0`.
            """
            return 4 if len(coerce_operand_list(a)) > 0 else 0

        @staticmethod
        def divide(a: BackendArg, b: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            """Divide `a` by `b`.

            Args:
                a: Numerator.
                b: Denominator.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Quotient array.
            """
            return ensure_ndarray(np.divide(ensure_ndarray(a), ensure_ndarray(b)))

        @staticmethod
        def array(a: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            """Wrap `a` as ndarray.

            Args:
                a: Scalar or array input.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Wrapped ndarray.
            """
            return ensure_ndarray(a)

    mock_mod = _ZeroFracStub()
    assert nan_module._np_zero_fraction(mock_mod, [0, 1, 0, 2]) == 0.5
    assert require_callable(numpy_eager_registry.get)("ZeroFraction")(mock_mod, [0, 1, 0, 2]) == 0.5

    assert np.isnan(ensure_ndarray(nan_module._np_zero_fraction(mock_mod, [])))

    # _xlogy
    x_log = np.array([0.0, 2.0])
    y_log = np.array([3.0, 4.0])
    assert np.allclose(nan_module._xlogy(x_log, y_log), [0.0, 2.0 * np.log(4.0)])

    # Nanmean and Nanmedian
    class _NanStatsStub(MockTestStub):
        """Stub backend module for `Nanmean` and `Nanmedian`."""

        @staticmethod
        def nanmean(a: BackendArg, axis: BackendArg = None, keepdims: BackendArg = False) -> float:
            """Return mock nanmean value.

            Args:
                a: Input array.
                axis: Reduction axis.
                keepdims: Keepdims flag.

            Returns:
                float: Constant `2.5`.
            """
            del a, axis, keepdims
            return 2.5

        @staticmethod
        def nanmedian(a: BackendArg, axis: BackendArg = None, keepdims: BackendArg = False) -> float:
            """Return mock nanmedian value.

            Args:
                a: Input array.
                axis: Reduction axis.
                keepdims: Keepdims flag.

            Returns:
                float: Constant `2.0`.
            """
            del a, axis, keepdims
            return 2.0

    mock_nan_mod = _NanStatsStub()
    arr_nan: NumpyTestArray = np.array([1.0, 2.0, np.nan, 4.0])
    assert nan_module._np_nanmean(mock_nan_mod, arr_nan) == 2.5
    assert require_callable(numpy_eager_registry.get)("Nanmean")(mock_nan_mod, arr_nan) == 2.5

    assert nan_module._np_nanmedian(mock_nan_mod, arr_nan) == 2.0
    assert require_callable(numpy_eager_registry.get)("Nanmedian")(mock_nan_mod, arr_nan) == 2.0
