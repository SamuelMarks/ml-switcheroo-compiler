"""Tests for test_numpy_eager_init."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import equal, execute_op, repeat, searchsorted, split, squeeze, stack, unstack

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_execute_op() -> None:
    """Test execute_op.

    Returns:
        None
    """
    res = typing.cast("EagerCallableProtocol", execute_op)(None, "Equal", typing.cast("EagerCallableProtocol", np.array)([1]), typing.cast("EagerCallableProtocol", np.array)([1]))
    assert res

    @global_eager_registry.register("DummyGlobalOp")
    def _dummy_global(np_mod: EagerValue, x: EagerValue) -> EagerValue:
        """Test function _dummy_global.

        Args:
            np_mod: Test parameter `np_mod`.
            x: Test parameter `x`.
        """
        return typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", 1)

    assert typing.cast("EagerCallableProtocol", execute_op)(None, "DummyGlobalOp", 1) == 2
    res = typing.cast("EagerCallableProtocol", execute_op)(None, "Add", typing.cast("EagerCallableProtocol", np.array)([1]), typing.cast("EagerCallableProtocol", np.array)([2]))
    assert typing.cast("EagerTensorProtocol", getattr(res, "tolist"))() == [3]
    res = typing.cast("EagerCallableProtocol", execute_op)(None, "Erf", typing.cast("EagerCallableProtocol", np.array)([0.0]))
    assert typing.cast("EagerTensorProtocol", getattr(res, "tolist"))() == [0.0]
    with pytest.raises(Exception, match=r".*"):
        typing.cast("EagerCallableProtocol", execute_op)(None, "NonExistentCrazyOpXYZ123")


def test_repeat() -> None:
    """Test repeat.

    Returns:
        None
    """
    arr = np.array([1, 2])
    res = repeat(np, arr, 2, axis=0)
    assert typing.cast("EagerTensorProtocol", getattr(res, "tolist"))() == [1, 1, 2, 2]


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
    res = split(np, arr, 2, axis=0)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), "tolist"))() == [1, 2]
    res2 = split(np, arr, 2, axis=0)
    assert typing.cast("EagerCallableProtocol", len)(res2) == 2


def test_squeeze() -> None:
    """Test squeeze.

    Returns:
        None
    """
    arr = np.array([[1], [2]])
    res = squeeze(np, arr, axis=1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "tolist"))() == [1, 2]
    res2 = squeeze(np, arr, axis=1)
    assert typing.cast("EagerTensorProtocol", getattr(res2, "tolist"))() == [1, 2]


def test_stack() -> None:
    """Test stack.

    Returns:
        None
    """
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    res = stack(np, [arr1, arr2], axis=1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res2 = stack(np, [arr1, arr2], axis=1)
    assert typing.cast("EagerTensorProtocol", getattr(res2, "shape")) == (2, 2)


def test_unstack() -> None:
    """Test unstack.

    Returns:
        None
    """
    arr = np.array([[1, 2], [3, 4]])
    res = unstack(np, arr, axis=1)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), "tolist"))() == [1, 3]
    res2 = unstack(np, [1, 2, 3], axis=0)
    assert typing.cast("EagerValue", res2) == (1, 2, 3)


def test_equal() -> None:
    """Test equal.

    Returns:
        None
    """
    res = equal(np, np.array([1]), np.array([1]))
    assert typing.cast("EagerTensorProtocol", getattr(res, "tolist"))() == [True]

    class IncompatibleType:
        """Test class IncompatibleType."""

        def __eq__(self, other: object) -> bool:
            """Test function __eq__.

            Args:
                other: Test parameter `other`.
            """
            return True

    res = equal(np, typing.cast("EagerValue", IncompatibleType()), typing.cast("EagerValue", IncompatibleType()))
    assert res


def test_equal_exception_fix() -> None:
    """Test function test_equal_exception_fix."""
    res = equal(np, [1], "string")
    assert not res


def test_execute_op_attribute_error_raise() -> None:
    """Test function test_execute_op_attribute_error_raise."""
    from ml_switcheroo_compiler.backends.numpy.eager import execute_op
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        typing.cast("EagerCallableProtocol", execute_op)(None, "OpThatThrowsAttr12345", 5)
