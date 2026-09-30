"""Tests for test_control_flow_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import control_flow as cf_module


def test_control_flow_ops() -> None:
    """Test control flow operations directly and through eager registry.

    Returns:
        None
    """
    # len(args) < 2
    res_single = cf_module._np_associative_scan(np, np.array([1, 2, 3]))
    assert np.array_equal(res_single, np.array([1, 2, 3]))

    # len(args) >= 2 with axis 0
    fn = lambda a, b: a + b
    elems = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    res_scan = cf_module._np_associative_scan(np, fn, elems, axis=0)
    expected_scan = np.array([[1, 2], [4, 6], [9, 12]])
    assert np.array_equal(res_scan, expected_scan)

    # len(args) >= 2 with axis 1
    elems_2d = np.array([[1, 2, 3], [4, 5, 6]])
    res_scan_axis1 = cf_module._np_associative_scan(np, fn, elems_2d, axis=1)
    expected_scan_axis1 = np.array([[1, 3, 6], [4, 9, 15]])
    assert np.array_equal(res_scan_axis1, expected_scan_axis1)

    # registry AssociativeScan
    reg_scan = numpy_eager_registry.get("AssociativeScan")
    reg_res = reg_scan(np, fn, elems, axis=0)
    assert np.array_equal(reg_res, expected_scan)

    # StopGradient
    x = np.array([1.0, 2.0])
    res_stop = cf_module._stop_gradient(np, x)
    assert np.array_equal(res_stop, x)

    reg_stop = numpy_eager_registry.get("StopGradient")
    assert np.array_equal(reg_stop(np, x), x)
