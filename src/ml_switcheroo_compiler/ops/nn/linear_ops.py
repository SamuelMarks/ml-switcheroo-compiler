# ruff: noqa
"""Linear transformations."""

from typing import Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add
from ml_switcheroo_compiler.ops.shape.frontend import swapaxes
from ml_switcheroo_compiler.ops.linalg.einsum_frontend import einsum
from ml_switcheroo_compiler.ops.linalg.matmul import matmul


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation to the incoming data: y = input @ weight.T + bias.

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


def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Applies a bilinear transformation to the incoming data.

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
