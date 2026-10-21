"""Unit tests for the shape system utilities in ml_switcheroo_compiler.

This module contains test cases to verify the correctness of shape broadcasting, matrix
multiplication shape inference, and axis normalization functions, including handling of
symbolic dimensions and error conditions.
"""

from __future__ import annotations

import typing
from collections import namedtuple

import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.ir.shape_system import ShapeTracker, SymbolicSolver, SymInt, broadcast_shapes, matmul_shape, normalize_axis
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_broadcast_shapes() -> None:
    """Test the broadcast shapes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert broadcast_shapes((2, 3), (2, 3)) == (2, 3)
    assert broadcast_shapes((3,), (2, 3)) == (2, 3)
    assert broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    assert broadcast_shapes((), (2, 3)) == (2, 3)
    with pytest.raises((ValueError, ShapeMismatchError)):
        broadcast_shapes((2, 3), (4, 3))
    assert broadcast_shapes(("B", 3), ("B", 3)) == ("B", 3)
    assert broadcast_shapes(("B", 3), ("T", 3)) == ("B", 3)
    from ml_switcheroo_compiler.ir.shape_system import SymbolicConstraintTracker

    tracker = SymbolicConstraintTracker()
    tracker.add_constraint("B", 4)
    tracker.add_constraint("T", 5)
    with pytest.raises((ValueError, ShapeMismatchError)):
        broadcast_shapes(("B", 3), ("T", 3), tracker=tracker)


def test_matmul_shape() -> None:
    """Test the matmul shape behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert matmul_shape((3,), (3,)) == ()
    assert matmul_shape((2, 3), (3, 4)) == (2, 4)
    assert matmul_shape((5, 2, 3), (5, 3, 4)) == (5, 2, 4)
    assert matmul_shape((5, 2, 3), (3, 4)) == (5, 2, 4)
    with pytest.raises((ValueError, ShapeMismatchError)):
        matmul_shape((), ())
    with pytest.raises((ValueError, ShapeMismatchError)):
        matmul_shape((3,), (4,))
    with pytest.raises((ValueError, ShapeMismatchError)):
        matmul_shape((2, 3), (4, 5))
    with pytest.raises((ValueError, ShapeMismatchError)):
        matmul_shape((5, 2, 3), (4, 3, 4))


def test_matmul_inner_mismatch_batched() -> None:
    """Test the matmul inner mismatch batched behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises((ValueError, ShapeMismatchError), match="Incompatible inner dimensions"):
        matmul_shape((5, 2, 3), (5, 4, 4))


def test_matmul_shape_1d() -> None:
    """Test the matmul shape 1d behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert matmul_shape((5,), (5,)) == ()
    assert matmul_shape((5,), (5, 4)) == (4,)
    assert matmul_shape((3, 5), (5,)) == (3,)


def test_normalize_axis() -> None:
    """Test the normalize axis behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert normalize_axis(0, 3) == 0
    assert normalize_axis(2, 3) == 2
    assert normalize_axis(-1, 3) == 2
    assert normalize_axis(-3, 3) == 0
    with pytest.raises((ValueError, ShapeMismatchError)):
        normalize_axis(3, 3)
    with pytest.raises((ValueError, ShapeMismatchError)):
        normalize_axis(-4, 3)
    assert normalize_axis((0, 1), 3) == (0, 1)
    assert normalize_axis((-1, -2), 3) == (2, 1)
    assert normalize_axis((0, -1), 3) == (0, 2)
    with pytest.raises((ValueError, ShapeMismatchError)):
        normalize_axis((0, 3), 3)
    with pytest.raises((ValueError, ShapeMismatchError)):
        normalize_axis((-4,), 3)
    with pytest.raises(TypeError):
        as_callable(normalize_axis)(1.5, 3)


def test_symint() -> None:
    """Test the symint behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = SymInt("A")
    b = SymInt("B")
    assert str(a + b) == "(A + B)"
    assert str(a + 2) == "(A + 2)"
    assert str(2 + a) == "(2 + A)"
    assert str(a - b) == "(A - B)"
    assert str(a - 2) == "(A - 2)"
    assert str(2 - a) == "(2 - A)"
    assert str(a * b) == "(A * B)"
    assert str(a * 2) == "(A * 2)"
    assert str(2 * a) == "(2 * A)"
    assert str(a // b) == "(A // B)"
    assert str(a // 2) == "(A // 2)"
    assert repr(a) == "SymInt(A)"
    assert a == SymInt("A")
    assert a != b
    assert a != "A"


def test_symbolic_solver() -> None:
    """Test the symbolic solver behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert SymbolicSolver.is_consistent(2, 2)
    assert not SymbolicSolver.is_consistent(2, 3)
    assert SymbolicSolver.is_consistent(SymInt("A"), SymInt("A"))
    assert not SymbolicSolver.is_consistent(SymInt("A"), SymInt("B"))


def test_shape_tracker() -> None:
    """Test the shape tracker behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    TensorSpec = namedtuple("TensorSpec", ["shape"])
    assert as_tensor(ShapeTracker.infer_elementwise([])) == ()
    t1 = TensorSpec(shape=(2, SymInt("A")))
    t2 = TensorSpec(shape=(SymInt("A"),))
    out = as_callable(ShapeTracker.infer_elementwise)([t1, t2])
    assert as_tensor(len(out)) == 2
    assert as_tensor(out[0]) == 2
    assert as_tensor(isinstance(as_tensor(out[1]), SymInt) and out[1].expr == "A")
    t3 = TensorSpec(shape=(SymInt("A"), 2))
    t4 = TensorSpec(shape=(2,))
    out_matmul = as_callable(ShapeTracker.infer_matmul)(t3, t4)
    assert as_tensor(len(out_matmul)) == 1
    assert as_tensor(isinstance(as_tensor(out_matmul[0]), SymInt) and out_matmul[0].expr == "A")
    t5 = TensorSpec(shape=(2, 3))
    t6 = TensorSpec(shape=(3, 4))
    out_matmul2 = as_callable(ShapeTracker.infer_matmul)(t5, t6)
    assert out_matmul2 == (2, 4)


def test_symint_hash() -> None:
    """Test the symint hash behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert hash(SymInt("a")) == hash("a")
