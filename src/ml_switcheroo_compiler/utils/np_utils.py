# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy-related utilities."""

from typing import Optional

from ml_switcheroo_compiler import ops


def to_categorical(x, num_classes: Optional[int] = None, dtype: str = "float32"):
    """Convert a class vector (integers) to binary class matrix.

    Args:
        x (object): The x parameter.
        num_classes (object): The num_classes parameter.
        dtype (str): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if num_classes is None:
        num_classes = int(ops.max(x).item() if hasattr(ops.max(x), "item") else ops.max(x)) + 1

    x_ops = ops.cast(x, "int32")
    indices = ops.arange(num_classes)
    x_expanded = ops.expand_dims(x_ops, -1)
    one_hot = ops.equal(x_expanded, indices)
    return ops.cast(one_hot, dtype)


def normalize(x, axis: int = -1, order: int = 2):
    """Normalize a tensor/array.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        order (int): The order parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    l2 = ops.sum(ops.square(x), axis=axis, keepdims=True)
    l2 = ops.maximum(ops.sqrt(l2), 1e-12)
    return ops.divide(x, l2)
