"""Module transform.py."""

from typing import Any

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Linalg transform ops."""

import ml_switcheroo_compiler.ops.binary as _math
import ml_switcheroo_compiler.ops.shape.joining as _joining
import ml_switcheroo_compiler.ops.shape.slicing as _slicing
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.creation.frontend_basic import array
from ml_switcheroo_compiler.ops.shape.frontend import reshape


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Any:  # type: ignore
    """Apply the Walsh-Hadamard Transform to the last dimension of the input tensor.

    Args:
        x (Tensor): The input tensor. Its last dimension must be a power of 2.
        scale (float): A scaling factor to apply to the output.

    Returns:
        Tensor: The transformed tensor.
    """
    h = 1
    n = x.shape[-1]
    res = x
    while h < n:
        res = reshape(res, list(res.shape[:-1]) + [-1, 2, h])

        x_part = _slicing.slice(res, axis=-2, start=0, end=1)
        y_part = _slicing.slice(res, axis=-2, start=1, end=2)

        res_plus = _math.add(x_part, y_part)
        res_minus = _math.subtract(x_part, y_part)

        stacked = _joining.concatenate([res_plus, res_minus], axis=-2)

        res = reshape(stacked, list(res.shape[:-3]) + [-1])
        h *= 2
    if scale != 1.0:
        res = _math.multiply(res, array([scale], dtype="float32"))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return res


__all__ = ["hadamard_transform"]
