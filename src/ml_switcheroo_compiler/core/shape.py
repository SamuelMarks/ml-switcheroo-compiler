"""Core shape utility functions."""


def _broadcast_dim(d1: int, d2: int, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> int:
    """Broadcast a single dimension."""
    if d1 == d2:
        return d1
    if d1 == 1:
        return d2
    if d2 == 1:
        return d1
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    raise ShapeMismatchError(f"Shapes {shape1} and {shape2} are incompatible.")


def broadcast_shapes(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    """Calculate the broadcasted shape of two tuples.

    Args:
        shape1: The first shape.
        shape2: The second shape.

    Returns:
        The broadcasted shape.

    Raises:
        ValueError: If the shapes are not compatible.
    """
    ndim = max(len(shape1), len(shape2))
    shape1_pad = (1,) * (ndim - len(shape1)) + tuple(shape1)
    shape2_pad = (1,) * (ndim - len(shape2)) + tuple(shape2)

    result = []
    for d1, d2 in zip(shape1_pad, shape2_pad):
        result.append(_broadcast_dim(d1, d2, shape1, shape2))
    return tuple(result)
