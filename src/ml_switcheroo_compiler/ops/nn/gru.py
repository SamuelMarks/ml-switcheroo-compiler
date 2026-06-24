"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.shape import split

from ml_switcheroo_compiler.ops.unary import tanh
from ml_switcheroo_compiler.nn.activations import sigmoid
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import matmul


def _compute_gru_gates(x_parts: tuple, r_parts: tuple, state: Tensor) -> Tensor:
    """Function docstring.

    Args:
        x_parts: Arg.
        r_parts: Arg.
        state: Arg.
    """
    x_z, x_r, x_h = x_parts
    recurrent_z, recurrent_r, recurrent_h = r_parts
    z = sigmoid(add(x_z, recurrent_z))
    r = sigmoid(add(x_r, recurrent_r))
    hh = tanh(add(x_h, multiply(r, recurrent_h)))
    return add(multiply(z, state), multiply(subtract(1.0, z), hh))


def gru_cell(
    inputs: Tensor,
    state: Tensor,
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Fused GRU cell math."""
    matrix_x = matmul(inputs, kernel)
    if bias is not None:
        matrix_x = add(matrix_x, bias)

    matrix_inner = matmul(state, recurrent_kernel)

    x_parts = split(matrix_x, 3, dim=-1)
    r_parts = split(matrix_inner, 3, dim=-1)

    h_new = _compute_gru_gates(x_parts, r_parts, state)
    return h_new, h_new
