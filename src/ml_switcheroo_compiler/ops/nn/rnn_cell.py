# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module rnn_cell.py."""

from typing import Any

"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.unary import tanh


def simple_rnn_cell(
    inputs: Tensor,  # type: ignore
    state: tuple[Tensor, ...],  # type: ignore
    kernel: Tensor,  # type: ignore
    recurrent_kernel: Tensor,  # type: ignore
    bias: Optional[Tensor] = None,  # type: ignore
) -> tuple[Tensor, tuple[Tensor, ...]]:  # type: ignore
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

    matrix_x = matmul(inputs, kernel)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(h_prev, recurrent_kernel)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    h_new = tanh(add(matrix_x, matrix_inner))

    return h_new, (h_new,)
