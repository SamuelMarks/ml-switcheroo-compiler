# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy-related utilities."""

from typing import Optional

from ml_switcheroo_compiler import ops


def to_categorical(x: object, num_classes: Optional[int] = None, dtype: str = "float32") -> object:
    """Convert a class vector (integers) to binary class matrix.

    Args:
        x (object): The x parameter.
        num_classes (object): The num_classes parameter.
        dtype (str): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if num_classes is None:
        num_classes: object = int(ops.max(x).item() if hasattr(ops.max(x), "item") else ops.max(x)) + 1

    x_ops: object = ops.cast(x, "int32")
    indices: object = ops.arange(num_classes)
    x_expanded: object = ops.expand_dims(x_ops, -1)
    one_hot: object = ops.equal(x_expanded, indices)
    return ops.cast(one_hot, dtype)


def normalize(x: object, axis: int = -1, order: int = 2) -> object:
    """Normalize a tensor/array.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        order (int): The order parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    l2: object = ops.sum(ops.square(x), axis=axis, keepdims=True)
    l2: object = ops.maximum(ops.sqrt(l2), 1e-12)
    return ops.divide(x, l2)
