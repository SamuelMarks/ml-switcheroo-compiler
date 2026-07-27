"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.unary import tanh


def simple_rnn_cell(
    inputs: Tensor,
    state: tuple[Tensor, ...],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Fused SimpleRNN cell math.

    Args:
        inputs (Tensor): The inputs.
        state (tuple[Tensor, ...]): The hidden state (usually a 1-element tuple).
        kernel (Tensor): The input weights.
        recurrent_kernel (Tensor): The recurrent weights.
        bias (Optional[Tensor]): The bias.

    Returns:
        tuple[Tensor, tuple[Tensor, ...]]: The output and new state.
    """
    h_prev = state[0]

    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(h_prev, recurrent_kernel)

    h_new = tanh(add(matrix_x, matrix_inner))

    return h_new, (h_new,)
