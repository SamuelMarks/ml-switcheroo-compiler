"""Linalg transform ops."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.binary as _math
import ml_switcheroo_compiler.ops.shape.joining as _joining
import ml_switcheroo_compiler.ops.shape.slicing as _slicing
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ops.creation.api_basic import array
from ml_switcheroo_compiler.ops.shape.api import reshape


def hadamard_transform(x: Tensor, scale: float = 1.0) -> OpValue:
    """Apply the Walsh-Hadamard Transform to the last dimension of the input tensor.

    Args:
        x (Tensor): The input tensor. Its last dimension must be a power of 2.
        scale (float): A scaling factor to apply to the output.

    Returns:
        Tensor: The transformed tensor.
    """
    from ml_switcheroo_compiler.core.shape import extract_shape

    h = 1
    n = extract_shape(x)[-1]
    res = x
    while h < n:
        res = reshape(res, list(res.shape[:-1]) + [-1, 2, h])

        x_part = _slicing.slice(res, axis=-2, start=0, end=1)
        y_part = _slicing.slice(res, axis=-2, start=1, end=2)

        res_plus = _math.add(x_part, y_part)
        res_minus = _math.subtract(x_part, y_part)

        stacked = _joining.concatenate([res_plus, res_minus], axis=-2)

        res = reshape(ensure_tensor(stacked), list(res.shape[:-3]) + [-1])
        h *= 2
    if scale != 1.0:
        from ml_switcheroo_compiler.core.dtype import DType

        res = _math.multiply(res, array([scale], dtype=DType.Float32))
    return res


__all__ = ["hadamard_transform"]
