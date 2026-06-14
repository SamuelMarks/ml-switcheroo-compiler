"""Core shape utility functions."""


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
    shape1 = (1,) * (ndim - len(shape1)) + tuple(shape1)
    shape2 = (1,) * (ndim - len(shape2)) + tuple(shape2)

    result = []
    for d1, d2 in zip(shape1, shape2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are incompatible.")
    return tuple(result)
