"""Test arg ops, assert, assign."""

import pytest
import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_arg_ops():
    config.eager_mode = True

    ArgSort = numpy_eager_registry.get("ArgSort")
    Argwhere = numpy_eager_registry.get("Argwhere")
    Argpartition = numpy_eager_registry.get("Argpartition")

    t = np.array([3, 1, 2])

    out = ArgSort(np, t)
    np.testing.assert_array_equal(out, np.array([1, 2, 0]))

    out = Argwhere(np, t > 1)
    np.testing.assert_array_equal(out, np.array([[0], [2]]))

    out = Argpartition(np, t, kth=1)
    assert out[1] == 2 or out[0] == 1

    # AsString and Assert
    AsString = numpy_eager_registry.get("AsString")
    out = AsString(np, t)
    assert out.dtype.kind in ("U", "S")

    Assert = numpy_eager_registry.get("Assert")
    Assert(np, np.array(True), data=["Everything is fine"])
    with pytest.raises(AssertionError):
        Assert(np, np.array(False), data=["Failed!"])

    # Assign
    Assign = numpy_eager_registry.get("Assign")
    t_ref = np.array([1, 2, 3])
    out = Assign(np, t_ref, np.array([4, 5, 6]))
    np.testing.assert_array_equal(out, np.array([4, 5, 6]))

    AssignAdd = numpy_eager_registry.get("AssignAdd")
    out_add = AssignAdd(np, t_ref, np.array([1, 1, 1]))
    np.testing.assert_array_equal(out_add, np.array([2, 3, 4]))

    AssignSub = numpy_eager_registry.get("AssignSub")
    out_sub = AssignSub(np, t_ref, np.array([1, 1, 1]))
    np.testing.assert_array_equal(out_sub, np.array([1, 2, 3]))
