"""RNN cell operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.unary import tanh


def simple_rnn_cell(
    inputs: Tensor,
    state: tuple[Tensor, ...],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Fused SimpleRNN cell math.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, ...]): Hidden state tuple (usually a 1-element tuple).
        kernel (Tensor): Input weights tensor.
        recurrent_kernel (Tensor): Recurrent state weights tensor.
        bias (Tensor | None): Optional bias tensor.

    Returns:
        tuple[Tensor, tuple[Tensor, ...]]: The computed output tensor and new state tuple.
    """
    h_prev = state[0]

    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(h_prev, recurrent_kernel)

    h_new = tanh(add(matrix_x, matrix_inner))

    return h_new, (h_new,)
