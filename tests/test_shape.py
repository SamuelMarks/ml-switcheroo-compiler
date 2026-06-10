"""Module docstring."""

import pytest
from ml_switcheroo.shape import broadcast_shapes, matmul_shape


def test_broadcast_shapes() -> None:
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


def test_matmul_shape() -> None:
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


def test_matmul_inner_mismatch_batched() -> None:
    """Docstring."""
    with pytest.raises(ValueError, match="Incompatible inner dimensions"):
        matmul_shape((5, 2, 3), (5, 4, 4))


def test_matmul_shape_1d() -> None:
    """Docstring."""
    assert matmul_shape((5,), (5,)) == ()
    assert matmul_shape((5,), (5, 4)) == (4,)
    assert matmul_shape((3, 5), (5,)) == (3,)


def test_normalize_axis() -> None:
    """Docstring."""
    from ml_switcheroo.shape import normalize_axis

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
