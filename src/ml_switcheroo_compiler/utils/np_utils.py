"""Numpy-related utilities."""

from ml_switcheroo_compiler import ops  # pragma: no cover


from typing import Optional  # pragma: no cover


def to_categorical(
    x: object, num_classes: Optional[int] = None, dtype: str = "float32"
) -> object:  # pragma: no cover
    """Converts a class vector (integers) to binary class matrix."""
    if num_classes is None:  # pragma: no cover
        if type(x).__module__ == "numpy":  # pragma: no cover
            np = __import__("numpy")  # pragma: no cover
            num_classes = int(np.max(x)) + 1  # pragma: no cover
        else:
            num_classes = int(ops.max(x)) + 1  # pragma: no cover

    if type(x).__module__ == "numpy":  # pragma: no cover
        np = __import__("numpy")  # pragma: no cover
        return np.eye(num_classes, dtype=dtype)[x]  # pragma: no cover

    x_ops = ops.cast(x, "int32")  # pragma: no cover
    indices = ops.arange(num_classes)  # pragma: no cover
    x_expanded = ops.expand_dims(x_ops, -1)  # pragma: no cover
    one_hot = ops.equal(x_expanded, indices)  # pragma: no cover
    return ops.cast(one_hot, dtype)  # pragma: no cover


def normalize(x: object, axis: int = -1, order: int = 2) -> object:  # pragma: no cover
    """Normalizes a tensor/array."""
    l2 = ops.sum(ops.square(x), axis=axis, keepdims=True)  # pragma: no cover
    l2 = ops.maximum(ops.sqrt(l2), 1e-12)  # pragma: no cover
    return ops.divide(x, l2)  # pragma: no cover
