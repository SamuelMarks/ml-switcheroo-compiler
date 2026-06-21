"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.ops.binary import add, multiply, subtract
from ml_switcheroo_compiler.ops.shape import split

from ml_switcheroo_compiler.ops.unary import tanh
from ml_switcheroo_compiler.nn.activations import sigmoid
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import matmul


def scan(
    f: object,
    init: tuple[Tensor, ...],
    xs: Tensor,
    length: Optional[int] = None,
    reverse: bool = False,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Scan loop construct.

    Args:
        f (object): The scan function.
        init (tuple[Tensor, ...]): The initial carry.
        xs (Tensor): The input sequence.
        length (Optional[int]): The length of the sequence.
        reverse (bool): Whether to reverse the sequence.

    Returns:
        tuple[tuple[Tensor, ...], Tensor]: The final carry and the stacked outputs.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import scan as cf_scan
    from ml_switcheroo_compiler.ops.shape import stack, unstack

    if config.eager_mode:
        xs_unstacked = unstack(xs, dim=0)

        if reverse:
            xs_unstacked = list(reversed(xs_unstacked))

        carry = init
        ys = []

        for x in xs_unstacked:
            carry, y = f(carry, x)
            ys.append(y)

        return carry, stack(ys, dim=0)
    else:
        if reverse:
            from ml_switcheroo_compiler.ops.shape import reverse as cf_reverse

            xs = cf_reverse(xs, (0,))

        carry, y = cf_scan(f, init, xs, length)

        return carry, y


def rnn(
    inputs: Tensor,
    initial_state: tuple[Tensor, ...],
    cell_fn: object,
    time_major: bool = False,
    go_backwards: bool = False,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Base recurrent loop evaluation.

    Args:
        inputs (Tensor): The input sequence.
        initial_state (tuple[Tensor, ...]): The initial states.
        cell_fn (object): The RNN cell function.
        time_major (bool): Whether inputs are time-major.
        go_backwards (bool): Whether to go backwards.

    Returns:
        tuple[Tensor, tuple[Tensor, ...]]: The output sequence and the final states.
    """
    from ml_switcheroo_compiler.ops.shape import permute

    if not time_major:
        # (batch, time, ...) -> (time, batch, ...)
        dims = list(range(len(inputs.shape)))
        dims[0], dims[1] = 1, 0
        inputs = permute(inputs, tuple(dims))

    def scan_fn(carry: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        out, new_carry = cell_fn(x, carry)
        return new_carry, out

    final_state, outputs = scan(scan_fn, initial_state, inputs, reverse=go_backwards)

    if not time_major:
        # (time, batch, ...) -> (batch, time, ...)
        dims = list(range(len(outputs.shape)))
        dims[0], dims[1] = 1, 0
        outputs = permute(outputs, tuple(dims))

    return outputs, final_state


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
    i, f, c_bar, o = split(z, 4, dim=-1)

    i = sigmoid(i)
    f = sigmoid(f)
    c_bar = tanh(c_bar)
    o = sigmoid(o)

    c_new = add(multiply(f, c), multiply(i, c_bar))
    h_new = multiply(o, tanh(c_new))

    return h_new, (h_new, c_new)


def _compute_gru_gates(x_parts: tuple, r_parts: tuple, state: Tensor) -> Tensor:
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
