"""Unit tests verifying shape broadcasting and matrix multiplication validation."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.shape import (
    _broadcast_dim,
    broadcast_shapes,
    validate_matrix_multiply_shapes,
)
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_broadcast_dim_equal() -> None:
    """Test broadcasting identical dimensions."""
    assert _broadcast_dim(4, 4, (4,), (4,)) == 4
    assert _broadcast_dim("?", "?", ("?",), ("?",)) == "?"


def test_broadcast_dim_ones_and_symbolic() -> None:
    """Test broadcasting unit and symbolic wildcard dimensions."""
    assert _broadcast_dim(1, 8, (1,), (8,)) == 8
    assert _broadcast_dim(8, 1, (8,), (1,)) == 8
    assert as_callable(_broadcast_dim)("?", 16, ("?",), (16,)) == 16
    assert as_callable(_broadcast_dim)(16, "?", (16,), ("?",)) == 16
    assert _broadcast_dim(-1, 32, (-1,), (32,)) == 32
    assert _broadcast_dim(32, -1, (32,), (-1,)) == 32


def test_broadcast_dim_mismatch() -> None:
    """Test error raised when dimensions cannot be broadcast together."""
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(4, 8, (4,), (8,))


def test_broadcast_shapes_success() -> None:
    """Test broadcasting various ranks, unit expansions, and symbolic shapes."""
    assert broadcast_shapes((), (3, 4)) == (3, 4)
    assert broadcast_shapes((1, 4), (5, 1)) == (5, 4)
    assert broadcast_shapes((2, 1, 6), (3, 6)) == (2, 3, 6)
    assert as_callable(broadcast_shapes)(("?", 1), (10, 4)) == (10, 4)


def test_broadcast_shapes_failure() -> None:
    """Test error raised when complete shape tuples are incompatible."""
    with pytest.raises(ShapeMismatchError):
        broadcast_shapes((3, 4), (3, 5))


def test_validate_matmul_shapes_strict_2d() -> None:
    """Test strict 2-D matrix multiplication validation rules."""
    assert validate_matrix_multiply_shapes((2, 3), (3, 4), strict_2d=True) == (2, 4)
    assert as_callable(validate_matrix_multiply_shapes)(("?", 3), (3, "?"), strict_2d=True) == ("?", "?")

    # Incompatible inner dimension
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3), (4, 5), strict_2d=True)

    # Incompatible rank (< 2 or > 2)
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3,), (3, 4), strict_2d=True)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3, 4), (2, 4, 5), strict_2d=True)


def test_validate_matmul_shapes_zero_rank() -> None:
    """Test error raised when an operand has rank 0."""
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((), (3, 4))
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3, 4), ())


def test_validate_matmul_shapes_1d_dot() -> None:
    """Test 1-D vector dot product shape inference."""
    assert validate_matrix_multiply_shapes((4,), (4,)) == ()
    assert as_callable(validate_matrix_multiply_shapes)(("?",), (4,)) == ()

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((4,), (5,))


def test_validate_matmul_shapes_vector_matrix() -> None:
    """Test 1-D vector with N-D matrix multiplication."""
    assert validate_matrix_multiply_shapes((3,), (3, 4)) == (4,)
    assert validate_matrix_multiply_shapes((3,), (2, 3, 4)) == (2, 4)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3,), (4, 5))


def test_validate_matmul_shapes_matrix_vector() -> None:
    """Test N-D matrix with 1-D vector multiplication."""
    assert validate_matrix_multiply_shapes((2, 3), (3,)) == (2,)
    assert validate_matrix_multiply_shapes((5, 2, 3), (3,)) == (5, 2)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3), (4,))


def test_validate_matmul_shapes_batched() -> None:
    """Test batched matrix multiplication with broadcasted batch dimensions."""
    assert validate_matrix_multiply_shapes((2, 3, 4), (4, 5)) == (2, 3, 5)
    assert validate_matrix_multiply_shapes((1, 3, 4), (2, 4, 5)) == (2, 3, 5)

    with pytest.raises(ShapeMismatchError):
        # Mismatched contracting dimension
        validate_matrix_multiply_shapes((2, 3, 4), (2, 5, 6))

    with pytest.raises(ShapeMismatchError):
        # Mismatched batch dimension
        validate_matrix_multiply_shapes((3, 2, 4), (4, 4, 5))


def test_shape_attr_and_cast_helpers() -> None:
    """Verify _has_attr, _get_attr, _set_attr, _del_attr, _obj_set_attr, _obj_del_attr, _obj_vars, _caller_locals, and _cast_val."""
    from ml_switcheroo_compiler.core.shape import (
        _caller_locals,
        _cast_val,
        _del_attr,
        _get_attr,
        _has_attr,
        _obj_del_attr,
        _obj_set_attr,
        _obj_vars,
        _set_attr,
    )

    class _Sample:
        """Sample test class."""

        pass

    inst = _Sample()
    assert _has_attr(inst, "missing") is False
    _set_attr(inst, "alpha", 10)
    assert _has_attr(inst, "alpha") is True
    assert _get_attr(inst, "alpha") == 10
    assert _get_attr(inst, "missing", 99) == 99
    assert "alpha" in _obj_vars(inst)
    _del_attr(inst, "alpha")
    assert _has_attr(inst, "alpha") is False

    _obj_set_attr(inst, "beta", 20)
    assert _get_attr(inst, "beta") == 20
    _obj_del_attr(inst, "beta")
    assert _has_attr(inst, "beta") is False

    _obj_set_attr(_Sample, "cls_attr", 30)
    assert _get_attr(_Sample, "cls_attr") == 30
    _obj_del_attr(_Sample, "cls_attr")
    assert _has_attr(_Sample, "cls_attr") is False

    local_marker = 42
    assert _caller_locals()["local_marker"] == local_marker
    assert _cast_val(int, 7) == 7
