"""Behavioral tests for NumPy eager control flow operations (`AssociativeScan`, `StopGradient`)."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager import control_flow_ops as cf_module
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_numeric, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _add_pair(lhs: BackendArg, rhs: BackendArg) -> BackendArg:
    """Compute the elementwise sum of two eager array operands.

    Args:
        lhs (BackendArg): Left-hand array operand.
        rhs (BackendArg): Right-hand array operand.

    Returns:
        BackendArg: Elementwise sum of `lhs` and `rhs`.
    """
    return coerce_numeric(lhs) + coerce_numeric(rhs)


def test_associative_scan_axis_and_single_arg_branches() -> None:
    """Verify `AssociativeScan` handles single-argument passthrough, 1D/list axis-0 scans, and 2D axis-1 scans."""
    single_input = np.array([1, 2, 3])
    res_single = cf_module._np_associative_scan(np, single_input)
    np.testing.assert_array_equal(ensure_ndarray(res_single), single_input)

    elems_1d = np.array([1, 2, 3, 4])
    res_1d = cf_module._np_associative_scan(np, _add_pair, elems_1d)
    np.testing.assert_array_equal(ensure_ndarray(res_1d), np.array([1, 3, 6, 10]))

    elems_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    expected_list_scan = np.array([[1, 2], [4, 6], [9, 12]])
    res_list = cf_module._np_associative_scan(np, _add_pair, elems_list, axis=0)
    np.testing.assert_array_equal(ensure_ndarray(res_list), expected_list_scan)

    reg_scan = numpy_eager_registry.get("AssociativeScan")
    np.testing.assert_array_equal(
        ensure_ndarray(require_callable(reg_scan)(np, _add_pair, elems_list, axis=0)), expected_list_scan
    )

    elems_2d = np.array([[1, 2, 3], [4, 5, 6]])
    expected_2d_axis1 = np.array([[1, 3, 6], [4, 9, 15]])
    res_2d = cf_module._np_associative_scan(np, _add_pair, elems_2d, axis=1)
    np.testing.assert_array_equal(ensure_ndarray(res_2d), expected_2d_axis1)


def test_stop_gradient_preserves_array_values() -> None:
    """Verify `StopGradient` returns an array copy with identical numerical values."""
    x = np.array([1.5, -2.5, 3.0], dtype=np.float32)
    res_direct = cf_module._stop_gradient(np, x)
    np.testing.assert_array_equal(ensure_ndarray(res_direct), x)

    reg_stop = numpy_eager_registry.get("StopGradient")
    np.testing.assert_array_equal(ensure_ndarray(require_callable(reg_stop)(np, x)), x)
