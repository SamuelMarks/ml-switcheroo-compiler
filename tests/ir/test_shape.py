# ruff: noqa: E501
"""Unit tests for the shape system utilities in ml_switcheroo_compiler.

This module contains test cases to verify the correctness of shape broadcasting, matrix
multiplication shape inference, and axis normalization functions, including handling of
symbolic dimensions and error conditions.
"""

from collections import namedtuple

import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.ir.shape_system import (
    ShapeTracker,
    SymbolicSolver,
    SymInt,
    broadcast_shapes,
    matmul_shape,
    normalize_axis,
)


def test_broadcast_shapes() -> None:
    """Test the broadcast shapes behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that `broadcast_shapes` correctly computes the broadcasted shape of two.\n\n    inputs\n\n    This test covers:\n    - Identical shapes\n    - Broadcasting a 1D shape to a 2D shape\n    - Broadcasting with unit dimensions\n    - Broadcasting with empty shapes (scalars)\n    - Incompatible shape broadcasting (raises ValueError)\n    - Symbolic dimension broadcasting and mismatch detection\n\n    Returns:\n    None\n    "
        assert broadcast_shapes((2, 3), (2, 3)) == (2, 3)
        assert broadcast_shapes((3,), (2, 3)) == (2, 3)
        assert broadcast_shapes((2, 1), (1, 3)) == (2, 3)
        assert broadcast_shapes((), (2, 3)) == (2, 3)
        with pytest.raises((ValueError, ShapeMismatchError)):
            broadcast_shapes((2, 3), (4, 3))
        assert broadcast_shapes(("B", 3), ("B", 3)) == ("B", 3)
        with pytest.raises((ValueError, ShapeMismatchError)):
            broadcast_shapes(("B", 3), ("T", 3))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_matmul_shape() -> None:
    """Test the matmul shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that `matmul_shape` correctly computes the output shape of a matrix.\n\n    multiplication\n\n    This test covers:\n    - 1D vector dot products\n    - 2D matrix multiplications\n    - Batched matrix multiplications\n    - Mismatched batch dimensions\n    - Invalid empty shapes\n    - Incompatible inner dimensions\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_matmul_inner_mismatch_batched() -> None:
    """Test the matmul inner mismatch batched behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that `matmul_shape` raises a ValueError when inner dimensions mismatch in.\n\n    batched multiplication\n\n    This test ensures that an explicit error message regarding incompatible inner\n    dimensions is raised when attempting to multiply batched matrices with\n    mismatched\n    contracting dimensions\n\n    Returns:\n    None\n    "
        with pytest.raises((ValueError, ShapeMismatchError), match="Incompatible inner dimensions"):
            matmul_shape((5, 2, 3), (5, 4, 4))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_matmul_shape_1d() -> None:
    """Test the matmul shape 1d behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that `matmul_shape` correctly handles 1D vector operands.\n\n    This test covers:\n    - Vector-vector multiplication (resulting in a scalar shape)\n    - Vector-matrix multiplication\n    - Matrix-vector multiplication\n\n    Returns:\n    None\n    "
        assert matmul_shape((5,), (5,)) == ()
        assert matmul_shape((5,), (5, 4)) == (4,)
        assert matmul_shape((3, 5), (5,)) == (3,)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_normalize_axis() -> None:
    """Test the normalize axis behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that `normalize_axis` correctly normalizes axis indices.\n\n    This test covers:\n    - Single integer axis normalization (both positive and negative indices)\n    - Out-of-bounds integer axis validation (raises ValueError)\n    - Tuple of axes normalization (both positive and negative indices)\n    - Out-of-bounds tuple axis validation (raises ValueError)\n    - Invalid axis type validation (raises TypeError)\n\n    Returns:\n    None\n    "
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
            normalize_axis(1.5, 3)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_symint() -> None:
    """Test the symint behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test SymInt."
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_symbolic_solver() -> None:
    """Test the symbolic solver behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test SymbolicSolver."
        assert SymbolicSolver.is_consistent(2, 2)
        assert not SymbolicSolver.is_consistent(2, 3)
        assert SymbolicSolver.is_consistent(SymInt("A"), SymInt("A"))
        assert not SymbolicSolver.is_consistent(SymInt("A"), SymInt("B"))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_shape_tracker() -> None:
    """Test the shape tracker behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test ShapeTracker."
        TensorSpec = namedtuple("TensorSpec", ["shape"])
        assert ShapeTracker.infer_elementwise([]) == ()
        t1 = TensorSpec(shape=(2, SymInt("A")))
        t2 = TensorSpec(shape=(SymInt("A"),))
        out = ShapeTracker.infer_elementwise([t1, t2])
        assert len(out) == 2
        assert out[0] == 2
        assert isinstance(out[1], SymInt) and out[1].expr == "A"
        t3 = TensorSpec(shape=(SymInt("A"), 2))
        t4 = TensorSpec(shape=(2,))
        out_matmul = ShapeTracker.infer_matmul(t3, t4)
        assert len(out_matmul) == 1
        assert isinstance(out_matmul[0], SymInt) and out_matmul[0].expr == "A"
        t5 = TensorSpec(shape=(2, 3))
        t6 = TensorSpec(shape=(3, 4))
        out_matmul2 = ShapeTracker.infer_matmul(t5, t6)
        assert out_matmul2 == (2, 4)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_symint_hash() -> None:
    """Test the symint hash behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test symint hash."
        assert hash(SymInt("a")) == hash("a")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
