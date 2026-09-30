"""Tests for test_math_nan_coverage."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_nan as nan_module


def test_math_nan_ops() -> None:
    """Test NaN-safe operations directly and through eager registry.

    Returns:
        None
    """
    x = np.array([4.0, 2.0, 0.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 0.0], dtype=np.float32)

    assert np.allclose(nan_module._np_divide_no_nan(np, x, y), [2.0, 0.5, 0.0])
    assert np.allclose(numpy_eager_registry.get("DivideNoNan")(np, x, y), [2.0, 0.5, 0.0])

    assert np.allclose(nan_module._np_multiply_no_nan(np, x, y), [8.0, 8.0, 0.0])
    assert np.allclose(numpy_eager_registry.get("MultiplyNoNan")(np, x, y), [8.0, 8.0, 0.0])

    assert np.allclose(nan_module._np_squared_difference(np, x, y), (x - y) ** 2)
    assert np.allclose(numpy_eager_registry.get("SquaredDifference")(np, x, y), (x - y) ** 2)

    assert np.allclose(nan_module._np_xdivy(np, x, y), [2.0, 0.5, 0.0])
    assert np.allclose(numpy_eager_registry.get("Xdivy")(np, x, y), [2.0, 0.5, 0.0])

    expected_xlog1py = np.array([4.0 * np.log1p(2.0), 2.0 * np.log1p(4.0), 0.0])
    assert np.allclose(nan_module._np_xlog1py(np, x, y), expected_xlog1py)
    assert np.allclose(numpy_eager_registry.get("Xlog1py")(np, x, y), expected_xlog1py)

    assert np.allclose(nan_module._np_reciprocal_no_nan(np, x), [0.25, 0.5, 0.0])
    assert np.allclose(numpy_eager_registry.get("ReciprocalNoNan")(np, x), [0.25, 0.5, 0.0])

    # ZeroFraction - non-empty and empty
    mock_mod = types.SimpleNamespace(
        sum=lambda a: 2,
        equal=lambda a, b: [True, False, True, False],
        size=lambda a: 4 if len(a) > 0 else 0,
        divide=lambda a, b: np.divide(a, b),
        array=lambda a: np.array(a),
    )
    assert nan_module._np_zero_fraction(mock_mod, [0, 1, 0, 2]) == 0.5
    assert numpy_eager_registry.get("ZeroFraction")(mock_mod, [0, 1, 0, 2]) == 0.5

    assert np.isnan(nan_module._np_zero_fraction(mock_mod, []))

    # _xlogy
    x_log = np.array([0.0, 2.0])
    y_log = np.array([3.0, 4.0])
    assert np.allclose(nan_module._xlogy(x_log, y_log), [0.0, 2.0 * np.log(4.0)])

    # Nanmean and Nanmedian
    mock_nan_mod = types.SimpleNamespace(
        nanmean=lambda a, axis=None, keepdims=False: 2.5,
        nanmedian=lambda a, axis=None, keepdims=False: 2.0,
    )
    arr_nan = np.array([1.0, 2.0, np.nan, 4.0])
    assert nan_module._np_nanmean(mock_nan_mod, arr_nan) == 2.5
    assert numpy_eager_registry.get("Nanmean")(mock_nan_mod, arr_nan) == 2.5

    assert nan_module._np_nanmedian(mock_nan_mod, arr_nan) == 2.0
    assert numpy_eager_registry.get("Nanmedian")(mock_nan_mod, arr_nan) == 2.0
