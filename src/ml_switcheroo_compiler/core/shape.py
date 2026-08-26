# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module shape.py."""

"""Core shape utility functions."""


def _broadcast_dim(d1: int, d2: int, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> int:
    """Broadcast a single dimension.

    Args:
        d1 (int): The d1 parameter.
        d2 (int): The d2 parameter.
        shape1 (tuple): The shape1 parameter.
        shape2 (tuple): The shape2 parameter.

    Returns:
        int: Result.

    Raises:
        ShapeMismatchError: An exception.
    """
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
        shape1 (tuple): The shape1 parameter.
        shape2 (tuple): The shape2 parameter.

    Returns:
        tuple: Result.
    """
    ndim = max(len(shape1), len(shape2))
    shape1_pad = (1,) * (ndim - len(shape1)) + tuple(shape1)
    shape2_pad = (1,) * (ndim - len(shape2)) + tuple(shape2)

    result = []
    for d1, d2 in zip(shape1_pad, shape2_pad):
        result.append(_broadcast_dim(d1, d2, shape1, shape2))
    return tuple(result)
