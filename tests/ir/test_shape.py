"""Unit tests for the shape system utilities in ml_switcheroo_compiler.

This module contains test cases to verify the correctness of shape broadcasting, matrix
multiplication shape inference, and axis normalization functions, including handling of
symbolic dimensions and error conditions.
"""

import pytest

from ml_switcheroo_compiler.ir.shape_system import broadcast_shapes, matmul_shape


def test_broadcast_shapes() -> None:
    """Verifies that `broadcast_shapes` correctly computes the broadcasted shape of two.

    inputs

    This test covers:
    - Identical shapes
    - Broadcasting a 1D shape to a 2D shape
    - Broadcasting with unit dimensions
    - Broadcasting with empty shapes (scalars)
    - Incompatible shape broadcasting (raises ValueError)
    - Symbolic dimension broadcasting and mismatch detection

    Returns:
    None
    """
    assert broadcast_shapes((2, 3), (2, 3)) == (2, 3)
    assert broadcast_shapes((3,), (2, 3)) == (2, 3)
    assert broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    assert broadcast_shapes((), (2, 3)) == (2, 3)

    with pytest.raises(ValueError):
        broadcast_shapes((2, 3), (4, 3))

    # Symbolic
    assert broadcast_shapes(("B", 3), ("B", 3)) == ("B", 3)
    with pytest.raises(ValueError):
        broadcast_shapes(("B", 3), ("T", 3))


def test_matmul_shape() -> None:
    """Verifies that `matmul_shape` correctly computes the output shape of a matrix.

    multiplication

    This test covers:
    - 1D vector dot products
    - 2D matrix multiplications
    - Batched matrix multiplications
    - Mismatched batch dimensions
    - Invalid empty shapes
    - Incompatible inner dimensions

    Returns:
    None
    """
    assert matmul_shape((3,), (3,)) == ()
    assert matmul_shape((2, 3), (3, 4)) == (2, 4)
    assert matmul_shape((5, 2, 3), (5, 3, 4)) == (5, 2, 4)
    assert matmul_shape((5, 2, 3), (3, 4)) == (5, 2, 4)

    with pytest.raises(ValueError):
        matmul_shape((), ())

    with pytest.raises(ValueError):
        matmul_shape((3,), (4,))

    with pytest.raises(ValueError):
        matmul_shape((2, 3), (4, 5))

    with pytest.raises(ValueError):
        matmul_shape((5, 2, 3), (4, 3, 4))


def test_matmul_inner_mismatch_batched() -> None:
    """Verifies that `matmul_shape` raises a ValueError when inner dimensions mismatch in.

    batched multiplication

    This test ensures that an explicit error message regarding incompatible inner
    dimensions is raised when attempting to multiply batched matrices with
    mismatched
    contracting dimensions

    Returns:
    None
    """
    with pytest.raises(ValueError, match="Incompatible inner dimensions"):
        matmul_shape((5, 2, 3), (5, 4, 4))


def test_matmul_shape_1d() -> None:
    """Verifies that `matmul_shape` correctly handles 1D vector operands.

    This test covers:
    - Vector-vector multiplication (resulting in a scalar shape)
    - Vector-matrix multiplication
    - Matrix-vector multiplication

    Returns:
    None
    """
    assert matmul_shape((5,), (5,)) == ()
    assert matmul_shape((5,), (5, 4)) == (4,)
    assert matmul_shape((3, 5), (5,)) == (3,)


def test_normalize_axis() -> None:
    """Verifies that `normalize_axis` correctly normalizes axis indices.

    This test covers:
    - Single integer axis normalization (both positive and negative indices)
    - Out-of-bounds integer axis validation (raises ValueError)
    - Tuple of axes normalization (both positive and negative indices)
    - Out-of-bounds tuple axis validation (raises ValueError)
    - Invalid axis type validation (raises TypeError)

    Returns:
    None
    """
    from ml_switcheroo_compiler.ir.shape_system import normalize_axis

    # int tests
    assert normalize_axis(0, 3) == 0
    assert normalize_axis(2, 3) == 2
    assert normalize_axis(-1, 3) == 2
    assert normalize_axis(-3, 3) == 0

    with pytest.raises(ValueError):
        normalize_axis(3, 3)

    with pytest.raises(ValueError):
        normalize_axis(-4, 3)

    # tuple tests
    assert normalize_axis((0, 1), 3) == (0, 1)
    assert normalize_axis((-1, -2), 3) == (2, 1)
    assert normalize_axis((0, -1), 3) == (0, 2)

    with pytest.raises(ValueError):
        normalize_axis((0, 3), 3)

    with pytest.raises(ValueError):
        normalize_axis((-4,), 3)

    # invalid type
    with pytest.raises(TypeError):
        normalize_axis(1.5, 3)


def test_symint() -> None:
    """Test SymInt."""
    from ml_switcheroo_compiler.ir.shape_system import SymInt

    a = SymInt("A")
    b = SymInt("B")

    # Add
    assert str(a + b) == "(A + B)"
    assert str(a + 2) == "(A + 2)"
    assert str(2 + a) == "(2 + A)"

    # Sub
    assert str(a - b) == "(A - B)"
    assert str(a - 2) == "(A - 2)"
    assert str(2 - a) == "(2 - A)"

    # Mul
    assert str(a * b) == "(A * B)"
    assert str(a * 2) == "(A * 2)"
    assert str(2 * a) == "(2 * A)"

    # Floordiv
    assert str(a // b) == "(A // B)"
    assert str(a // 2) == "(A // 2)"

    # Eq / repr
    assert repr(a) == "SymInt(A)"
    assert a == SymInt("A")
    assert a != b
    assert a != "A"


def test_symbolic_solver() -> None:
    """Test SymbolicSolver."""
    from ml_switcheroo_compiler.ir.shape_system import SymbolicSolver, SymInt

    assert SymbolicSolver.is_consistent(2, 2)
    assert not SymbolicSolver.is_consistent(2, 3)
    assert SymbolicSolver.is_consistent(SymInt("A"), SymInt("A"))
    assert not SymbolicSolver.is_consistent(SymInt("A"), SymInt("B"))


def test_shape_tracker() -> None:
    """Test ShapeTracker."""
    from ml_switcheroo_compiler.ir.shape_system import ShapeTracker, SymInt
    from collections import namedtuple

    TensorSpec = namedtuple("TensorSpec", ["shape"])

    # infer_elementwise
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
