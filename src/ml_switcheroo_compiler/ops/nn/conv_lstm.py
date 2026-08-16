# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""RNN operations."""

from typing import Any, Optional

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3, MAGIC_VAL_4, MAGIC_VAL_5
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.nn.conv import (
    conv1d,
    conv2d,
    conv3d,
)
from ml_switcheroo_compiler.ops.shape import split
from ml_switcheroo_compiler.ops.unary import tanh

from .rnn_utils import ConvLSTMConfig, RNNWeights


def conv_lstm_cell(
    inputs: Tensor,  # type: ignore
    state: tuple[Tensor, Tensor],  # type: ignore
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # type: ignore
    """Provide generic Convolutional LSTM cell.

    Args:
        inputs (Tensor): The inputs parameter.
        state (tuple): The state parameter.
        weights (RNNWeights): The weights parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.

    Raises:
        ValueError: An exception.
    """
    naxis = len(inputs.shape)
    if naxis == MAGIC_VAL_3:
        return conv1d_lstm_cell(inputs, state, weights, config)
    elif naxis == MAGIC_VAL_4:
        return conv2d_lstm_cell(inputs, state, weights, config)
    elif naxis == MAGIC_VAL_5:
        return conv3d_lstm_cell(inputs, state, weights, config)
    else:
        raise ValueError(f"Unsupported input dimension for conv_lstm_cell: {naxis}. Expected 3, 4, or 5.")


def _apply_conv_lstm_gates(
    x_conv: Tensor,  # type: ignore
    h_conv: Tensor,  # type: ignore
    state: tuple[Tensor, Tensor],  # type: ignore
    weights: RNNWeights,
    data_format: str,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # type: ignore
    """Apply LSTM gates.

    Args:
        x_conv (Tensor): The x_conv parameter.
        h_conv (Tensor): The h_conv parameter.
        state (tuple): The state parameter.
        weights (RNNWeights): The weights parameter.
        data_format (str): The data_format parameter.

    Returns:
        tuple: Result.
    """
    h_prev, c_prev = state
    gates = add(x_conv, h_conv)
    if weights.bias is not None:
        gates = add(gates, weights.bias)

    axis_val = -1 if data_format == "channels_last" else 1
    i, f, c, o = split(gates, 4, axis=axis_val)

    i = _sigmoid(i)
    f = _sigmoid(f)
    c = tanh(c)
    o = _sigmoid(o)

    new_c = add(multiply(f, c_prev), multiply(i, c))
    new_h = multiply(o, tanh(new_c))

    return new_h, (new_h, new_c)


def conv1d_lstm_cell(
    inputs: Tensor,  # type: ignore
    state: tuple[Tensor, Tensor],  # type: ignore
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # type: ignore
    """1D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state

    conf = config if config is not None else ConvLSTMConfig()
    x_conv = conv1d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )
    h_conv = conv1d(
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)


def conv2d_lstm_cell(
    inputs: Tensor,  # type: ignore
    state: tuple[Tensor, Tensor],  # type: ignore
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # type: ignore
    """2D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state

    conf = config if config is not None else ConvLSTMConfig()
    x_conv = conv2d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )
    h_conv = conv2d(
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)


def conv3d_lstm_cell(
    inputs: Tensor,  # type: ignore
    state: tuple[Tensor, Tensor],  # type: ignore
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # type: ignore
    """3D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state

    conf = config if config is not None else ConvLSTMConfig()
    x_conv = conv3d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )
    h_conv = conv3d(
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)


def _sigmoid(x: Any) -> Any:
    """Sigmoid.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.nn.activations import sigmoid as s

    return s(x)
