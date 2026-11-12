"""One-hot categorical encoding and L-p tensor normalization utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.shape import TensorOrScalar


@typing.runtime_checkable
class _ScalarItemProtocol(typing.Protocol):
    """Protocol for scalar tensors exposing `.item()`."""

    def item(self) -> int | float:
        """Return scalar value.

        Returns:
            int | float: Scalar numeric value.
        """
        ...


def _extract_scalar_int(val: TensorOrScalar) -> int:
    """Extract an integer scalar from a tensor or numeric scalar.

    Args:
        val (TensorOrScalar): Input scalar or 0-D tensor.

    Returns:
        int: Extracted integer value.
    """
    if isinstance(val, _ScalarItemProtocol):
        return int(val.item())
    return int(val) if isinstance(val, (int, float)) else 0


def to_categorical(x: TensorOrScalar, num_classes: int | None = None, dtype: str = "float32") -> TensorOrScalar:
    """Convert an integer class index tensor into a one-hot binary class matrix.

    Args:
        x (TensorOrScalar): Input tensor or sequence of integer class indices.
        num_classes (Optional[int]): Total number of classes. Inferred as `max(x) + 1` if None.
        dtype (str): Output data type string (default `'float32'`).

    Returns:
        TensorOrScalar: One-hot encoded tensor with trailing dimension `num_classes`.
    """
    if num_classes is None:
        max_val = ops.max(x)
        num_classes = _extract_scalar_int(max_val) + 1

    op_cast = ops.cast
    x_ops = op_cast(x, "int32")
    indices = ops.arange(num_classes)
    x_expanded = ops.expand_dims(x_ops, -1)
    one_hot = ops.equal(x_expanded, indices)
    return op_cast(one_hot, dtype)


def normalize(x: TensorOrScalar, axis: int = -1, order: int | float = 2) -> TensorOrScalar:
    """Normalize a tensor along `axis` to unit L-p norm for arbitrary `order`.

    Args:
        x (TensorOrScalar): Input tensor to normalize.
        axis (int): Axis along which to compute the norm.
        order (int | float): Norm order (`0`, `1`, `2`, `float('inf')`, or general `p > 0`).

    Returns:
        TensorOrScalar: Normalized tensor of the same shape as `x`.
    """
    if order == 2:
        norm_val = ops.sqrt(ops.sum(ops.square(x), axis=axis, keepdims=True))
    elif order == 1:
        norm_val = ops.sum(ops.abs(x), axis=axis, keepdims=True)
    elif order == 0:
        op_cast = ops.cast
        nonzero = op_cast(ops.not_equal(x, 0), "float32")
        norm_val = ops.sum(nonzero, axis=axis, keepdims=True)
    elif order == float("inf"):
        norm_val = ops.max(ops.abs(x), axis=axis, keepdims=True)
    else:
        powered = ops.power(ops.abs(x), order)
        summed = ops.sum(powered, axis=axis, keepdims=True)
        norm_val = ops.power(summed, 1.0 / float(order))
    denom = ops.maximum(norm_val, 1e-12)
    return ops.divide(x, denom)
