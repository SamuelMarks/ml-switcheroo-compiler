"""Module linear_ops.py."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Linear transformations."""

from typing import Any, Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add
from ml_switcheroo_compiler.ops.linalg.einsum_frontend import einsum
from ml_switcheroo_compiler.ops.linalg.matmul import matmul
from ml_switcheroo_compiler.ops.shape.frontend import swapaxes


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    """Apply a linear transformation to the incoming data: y = input @ weight.T + bias.

    Args:
        input: Incoming data.
        weight: Weights of the transformation. Expected shape is (out_features, in_features).
        bias: Optional bias to add. Expected shape is (out_features,).

    Returns:
        The transformed tensor.
    """
    out = matmul(input, swapaxes(weight, -1, -2))
    if bias is not None:
        out = add(out, bias)
    return out


def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    """Apply a bilinear transformation to the incoming data.

    y = input1 @ weight @ input2 + bias (broadcasting over batch dims).

    Args:
        input1: First incoming data. Shape `(..., in1_features)`.
        input2: Second incoming data. Shape `(..., in2_features)`.
        weight: Weights of the transformation. Shape `(out_features, in1_features, in2_features)`.
        bias: Optional bias to add. Shape `(out_features,)`.

    Returns:
        The transformed tensor. Shape `(..., out_features)`.
    """
    # Computes sum_i sum_k input1[..., i] * weight[j, i, k] * input2[..., k] -> out[..., j]
    out = einsum("...i,jik,...k->...j", input1, weight, input2)
    if bias is not None:
        out = add(out, bias)
    return out
