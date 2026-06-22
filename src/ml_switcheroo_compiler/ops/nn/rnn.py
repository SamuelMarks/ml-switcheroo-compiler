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
    unroll: bool = False,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Scan loop construct.

    Args:
        f (object): The scan function.
        init (tuple[Tensor, ...]): The initial carry.
        xs (Tensor): The input sequence.
        length (Optional[int]): The length of the sequence.
        reverse (bool): Whether to reverse the sequence.
        unroll (bool): Whether to unroll the loop.

    Returns:
        tuple[tuple[Tensor, ...], Tensor]: The final carry and the stacked outputs.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import scan as cf_scan
    from ml_switcheroo_compiler.ops.shape import stack, unstack

    if config.eager_mode or unroll:
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


def bidirectional(
    forward_inputs: Tensor,
    backward_inputs: Tensor,
    forward_initial_state: tuple[Tensor, ...],
    backward_initial_state: tuple[Tensor, ...],
    cell_fn: object,
    merge_mode: str = "concat",
    time_major: bool = False,
    unroll: bool = False,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Bidirectional RNN wrapper.

    Args:
        forward_inputs (Tensor): The forward input sequence.
        backward_inputs (Tensor): The backward input sequence.
        forward_initial_state (tuple[Tensor, ...]): Initial states for forward direction.
        backward_initial_state (tuple[Tensor, ...]): Initial states for backward direction.
        cell_fn (object): The RNN cell function.
        merge_mode (str): How to merge outputs ('concat', 'sum', 'mul', 'ave', or None).
        time_major (bool): Whether inputs are time-major.
        unroll (bool): Whether to unroll the loop.

    Returns:
        tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
            Merged output sequence, forward final states, backward final states.
    """
    from ml_switcheroo_compiler.ops.shape import concatenate
    from ml_switcheroo_compiler.ops.binary import add, multiply

    forward_out, forward_state = rnn(
        forward_inputs,
        forward_initial_state,
        cell_fn,
        time_major=time_major,
        unroll=unroll,
        go_backwards=False,
    )

    backward_out, backward_state = rnn(
        backward_inputs,
        backward_initial_state,
        cell_fn,
        time_major=time_major,
        unroll=unroll,
        go_backwards=True,
    )

    if merge_mode == "concat":
        merged_out = concatenate([forward_out, backward_out], dim=-1)
    elif merge_mode == "sum":
        merged_out = add(forward_out, backward_out)
    elif merge_mode == "mul":
        merged_out = multiply(forward_out, backward_out)
    elif merge_mode == "ave":
        merged_out = multiply(add(forward_out, backward_out), 0.5)
    else:
        # None
        merged_out = (forward_out, backward_out)

    return merged_out, forward_state, backward_state


def rnn(
    inputs: Tensor,
    initial_state: tuple[Tensor, ...],
    cell_fn: object,
    time_major: bool = False,
    go_backwards: bool = False,
    unroll: bool = False,
    return_all_outputs: bool = True,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Base recurrent loop evaluation.

    Args:
        inputs (Tensor): The input sequence.
        initial_state (tuple[Tensor, ...]): The initial states.
        cell_fn (object): The RNN cell function.
        time_major (bool): Whether inputs are time-major.
        go_backwards (bool): Whether to go backwards.
        unroll (bool): Whether to unroll the loop.
        return_all_outputs (bool): Whether to return all outputs or just the last.

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

    final_state, outputs = scan(scan_fn, initial_state, inputs, reverse=go_backwards, unroll=unroll)
    if not return_all_outputs:
        outputs = outputs[-1] if time_major else outputs[:, -1]

    if not time_major:
        # (time, batch, ...) -> (batch, time, ...)
        dims = list(range(len(outputs.shape)))
        dims[0], dims[1] = 1, 0
        outputs = permute(outputs, tuple(dims))

    return outputs, final_state


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


def conv_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
    strides: int = 1,
    padding: str = "SAME",
    data_format: str = "channels_last",
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Generic Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        kernel (Tensor): Convolution kernel.
        recurrent_kernel (Tensor): Recurrent kernel.
        bias (Optional[Tensor]): Bias tensor.
        strides (int): Convolution strides.
        padding (str): Padding mode.
        data_format (str): Data format.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    ndim = len(inputs.shape)
    if ndim == 3:
        return conv1d_lstm_cell(
            inputs, state, kernel, recurrent_kernel, bias, strides, padding, data_format
        )
    elif ndim == 4:
        return conv2d_lstm_cell(
            inputs, state, kernel, recurrent_kernel, bias, strides, padding, data_format
        )
    elif ndim == 5:
        return conv3d_lstm_cell(
            inputs, state, kernel, recurrent_kernel, bias, strides, padding, data_format
        )
    else:
        raise ValueError(
            f"Unsupported input dimension for conv_lstm_cell: {ndim}. Expected 3, 4, or 5."
        )


def conv1d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
    strides: int = 1,
    padding: str = "SAME",
    data_format: str = "channels_last",
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """1D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        kernel (Tensor): Convolution kernel.
        recurrent_kernel (Tensor): Recurrent kernel.
        bias (Optional[Tensor]): Bias tensor.
        strides (int): Convolution strides.
        padding (str): Padding mode.
        data_format (str): Data format.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    from ml_switcheroo_compiler.ops.nn.conv import conv1d

    h_prev, c_prev = state

    x_conv = conv1d(inputs, kernel, strides=strides, padding=padding, data_format=data_format)
    h_conv = conv1d(
        h_prev, recurrent_kernel, strides=strides, padding=padding, data_format=data_format
    )

    gates = add(x_conv, h_conv)
    if bias is not None:
        gates = add(gates, bias)

    if data_format == "channels_last":
        i, f, c, o = split(gates, 4, dim=-1)
    else:
        i, f, c, o = split(gates, 4, dim=1)

    i = sigmoid(i)
    f = sigmoid(f)
    c = tanh(c)
    o = sigmoid(o)

    new_c = add(multiply(f, c_prev), multiply(i, c))
    new_h = multiply(o, tanh(new_c))

    return new_h, (new_h, new_c)


def conv2d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
    strides: int = 1,
    padding: str = "SAME",
    data_format: str = "channels_last",
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """2D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        kernel (Tensor): Convolution kernel.
        recurrent_kernel (Tensor): Recurrent kernel.
        bias (Optional[Tensor]): Bias tensor.
        strides (int): Convolution strides.
        padding (str): Padding mode.
        data_format (str): Data format.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    from ml_switcheroo_compiler.ops.nn.conv import conv2d

    h_prev, c_prev = state

    x_conv = conv2d(inputs, kernel, strides=strides, padding=padding, data_format=data_format)
    h_conv = conv2d(
        h_prev, recurrent_kernel, strides=strides, padding=padding, data_format=data_format
    )

    gates = add(x_conv, h_conv)
    if bias is not None:
        gates = add(gates, bias)

    if data_format == "channels_last":
        i, f, c, o = split(gates, 4, dim=-1)
    else:
        i, f, c, o = split(gates, 4, dim=1)

    i = sigmoid(i)
    f = sigmoid(f)
    c = tanh(c)
    o = sigmoid(o)

    new_c = add(multiply(f, c_prev), multiply(i, c))
    new_h = multiply(o, tanh(new_c))

    return new_h, (new_h, new_c)


def conv3d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    kernel: Tensor,
    recurrent_kernel: Tensor,
    bias: Optional[Tensor] = None,
    strides: int = 1,
    padding: str = "SAME",
    data_format: str = "channels_last",
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """3D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        kernel (Tensor): Convolution kernel.
        recurrent_kernel (Tensor): Recurrent kernel.
        bias (Optional[Tensor]): Bias tensor.
        strides (int): Convolution strides.
        padding (str): Padding mode.
        data_format (str): Data format.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    from ml_switcheroo_compiler.ops.nn.conv import conv3d

    h_prev, c_prev = state

    x_conv = conv3d(inputs, kernel, strides=strides, padding=padding, data_format=data_format)
    h_conv = conv3d(
        h_prev, recurrent_kernel, strides=strides, padding=padding, data_format=data_format
    )

    gates = add(x_conv, h_conv)
    if bias is not None:
        gates = add(gates, bias)

    if data_format == "channels_last":
        i, f, c, o = split(gates, 4, dim=-1)
    else:
        i, f, c, o = split(gates, 4, dim=1)

    i = sigmoid(i)
    f = sigmoid(f)
    c = tanh(c)
    o = sigmoid(o)

    new_c = add(multiply(f, c_prev), multiply(i, c))
    new_h = multiply(o, tanh(new_c))

    return new_h, (new_h, new_c)
