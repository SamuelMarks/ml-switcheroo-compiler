"""Numpy-related utilities."""

from typing import Optional

from ml_switcheroo_compiler import ops


def to_categorical(x: object, num_classes: Optional[int] = None, dtype: str = "float32") -> object:
    """Converts a class vector (integers) to binary class matrix."""
    if num_classes is None:
        num_classes = int(ops.max(x).item() if hasattr(ops.max(x), "item") else ops.max(x)) + 1

    x_ops = ops.cast(x, "int32")
    indices = ops.arange(num_classes)
    x_expanded = ops.expand_dims(x_ops, -1)
    one_hot = ops.equal(x_expanded, indices)
    return ops.cast(one_hot, dtype)


def normalize(x: object, axis: int = -1, order: int = 2) -> object:
    """Normalizes a tensor/array."""
    l2 = ops.sum(ops.square(x), axis=axis, keepdims=True)
    l2 = ops.maximum(ops.sqrt(l2), 1e-12)
    return ops.divide(x, l2)
