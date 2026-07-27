# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.__init__ import equal, execute_op, repeat, searchsorted, split, squeeze, stack, unstack

"Tests for numpy eager init functions."


def test_execute_op() -> None:
    """Test execute_op.

    Returns:
        None
    """
    res = execute_op(None, "Equal", np.array([1]), np.array([1]))
    assert res

    @global_eager_registry.register("DummyGlobalOp")
    def _dummy_global(np_mod, x):
        return x + 1

    assert execute_op(None, "DummyGlobalOp", 1) == 2
    res = execute_op(None, "Add", np.array([1]), np.array([2]))
    assert res.tolist() == [3]
    res = execute_op(None, "Erf", np.array([0.0]))
    assert res.tolist() == [0.0]
    with pytest.raises(Exception):
        execute_op(None, "NonExistentCrazyOpXYZ123")


def test_repeat() -> None:
    """Test repeat.

    Returns:
        None
    """
    arr = np.array([1, 2])
    res = repeat(np, arr, 2, dim=0)
    assert res.tolist() == [1, 1, 2, 2]


def test_searchsorted() -> None:
    """Test searchsorted.

    Returns:
        None
    """
    arr = np.array([1, 3, 5])
    res = searchsorted(np, arr, 2)
    assert res == 1


def test_split() -> None:
    """Test split.

    Returns:
        None
    """
    arr = np.array([1, 2, 3, 4])
    res = split(np, arr, 2, dim=0)
    assert len(res) == 2
    assert res[0].tolist() == [1, 2]
    res2 = split(np, arr, 2, axis=0)
    assert len(res2) == 2


def test_squeeze() -> None:
    """Test squeeze.

    Returns:
        None
    """
    arr = np.array([[1], [2]])
    res = squeeze(np, arr, dim=1)
    assert res.tolist() == [1, 2]
    res2 = squeeze(np, arr, axis=1)
    assert res2.tolist() == [1, 2]


def test_stack() -> None:
    """Test stack.

    Returns:
        None
    """
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    res = stack(np, [arr1, arr2], dim=1)
    assert res.shape == (2, 2)
    res2 = stack(np, [arr1, arr2], axis=1)
    assert res2.shape == (2, 2)


def test_unstack() -> None:
    """Test unstack.

    Returns:
        None
    """
    arr = np.array([[1, 2], [3, 4]])
    res = unstack(np, arr, dim=1)
    assert len(res) == 2
    assert res[0].tolist() == [1, 3]
    res2 = unstack(np, [1, 2, 3], axis=0)
    assert res2 == (1, 2, 3)


def test_equal() -> None:
    """Test equal.

    Returns:
        None
    """
    res = equal(np, np.array([1]), np.array([1]))
    assert res.tolist() == [True]

    class IncompatibleType:
        def __eq__(self, other):
            return True

    res = equal(np, IncompatibleType(), IncompatibleType())
    assert res


def test_equal_exception_fix() -> None:
    res = equal(np, [1], "string")
    assert not res
