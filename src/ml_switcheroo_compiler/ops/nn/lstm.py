# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""RNN operations."""

from typing import Any, Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.shape.splitting import split
from ml_switcheroo_compiler.ops.unary import tanh


def lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Fused LSTM cell math.

    Args:
        inputs (Tensor): The inputs.
        state (tuple[Tensor, Tensor]): The hidden state and cell state.
        kernel (Tensor): The input weights.
        recurrent_kernel (Tensor): The recurrent weights.
        bias (Optional[Tensor]): The bias.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The output and new state (h, c).
    """
    h, c = state

    # Check dimensions
    z = add(matmul(inputs, kernel), matmul(h, recurrent_kernel))
    if bias is not None:
        z = add(z, bias)

    # Split z into i, f, c_bar, o
    i, f, c_bar, o = split(z, 4, axis=-1)

    i = _sigmoid(i)
    f = _sigmoid(f)
    c_bar = tanh(c_bar)
    o = _sigmoid(o)

    c_new = add(multiply(f, c), multiply(i, c_bar))
    h_new = multiply(o, tanh(c_new))

    return h_new, (h_new, c_new)


def _sigmoid(x):
    """Evaluate _sigmoid operation.

    Args:
        x (Any): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.nn.activations import sigmoid as s

    return s(x)
