"""Module docstring."""

import pytest
from ml_switcheroo_compiler.shape import broadcast_shapes, matmul_shape


def test_broadcast_shapes():
    """Docstring."""
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


def test_matmul_shape():
    """Docstring."""
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


def test_matmul_inner_mismatch_batched():
    """Docstring."""
    with pytest.raises(ValueError, match="Incompatible inner dimensions"):
        matmul_shape((5, 2, 3), (5, 4, 4))


def test_matmul_shape_1d():
    """Docstring."""
    assert matmul_shape((5,), (5,)) == ()
    assert matmul_shape((5,), (5, 4)) == (4,)
    assert matmul_shape((3, 5), (5,)) == (3,)
