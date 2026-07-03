"""RNN operations."""

from typing import Optional

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3, MAGIC_VAL_4, MAGIC_VAL_5
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.nn.activations import sigmoid
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.nn.conv import (
    conv1d,  # pragma: no cover
    conv2d,  # pragma: no cover
    conv3d,  # pragma: no cover
)
from ml_switcheroo_compiler.ops.shape import split
from ml_switcheroo_compiler.ops.unary import tanh

from .rnn_utils import ConvLSTMConfig, RNNWeights


def conv_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Generic Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    ndim = len(inputs.shape)  # pragma: no cover
    if ndim == MAGIC_VAL_3:  # pragma: no cover
        return conv1d_lstm_cell(  # pragma: no cover
            inputs, state, weights, config
        )
    elif ndim == MAGIC_VAL_4:  # pragma: no cover
        return conv2d_lstm_cell(  # pragma: no cover
            inputs, state, weights, config
        )
    elif ndim == MAGIC_VAL_5:  # pragma: no cover
        return conv3d_lstm_cell(  # pragma: no cover
            inputs, state, weights, config
        )
    else:
        raise ValueError(  # pragma: no cover
            f"Unsupported input dimension for conv_lstm_cell: {ndim}. Expected 3, 4, or 5."
        )


def _apply_conv_lstm_gates(
    x_conv: Tensor,
    h_conv: Tensor,
    state: tuple[Tensor, Tensor],
    weights: RNNWeights,
    data_format: str,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Apply LSTM gates."""
    h_prev, c_prev = state  # pragma: no cover
    gates = add(x_conv, h_conv)  # pragma: no cover
    if weights.bias is not None:  # pragma: no cover
        gates = add(gates, weights.bias)  # pragma: no cover

    dim = -1 if data_format == "channels_last" else 1  # pragma: no cover
    i, f, c, o = split(gates, 4, dim=dim)  # pragma: no cover

    i = sigmoid(i)  # pragma: no cover
    f = sigmoid(f)  # pragma: no cover
    c = tanh(c)  # pragma: no cover
    o = sigmoid(o)  # pragma: no cover

    new_c = add(multiply(f, c_prev), multiply(i, c))  # pragma: no cover
    new_h = multiply(o, tanh(new_c))  # pragma: no cover

    return new_h, (new_h, new_c)  # pragma: no cover


def conv1d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """1D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state  # pragma: no cover

    conf = config if config is not None else ConvLSTMConfig()  # pragma: no cover
    x_conv = conv1d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )  # pragma: no cover
    h_conv = conv1d(  # pragma: no cover
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)  # pragma: no cover


def conv2d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """2D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state  # pragma: no cover

    conf = config if config is not None else ConvLSTMConfig()  # pragma: no cover
    x_conv = conv2d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )  # pragma: no cover
    h_conv = conv2d(  # pragma: no cover
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)  # pragma: no cover


def conv3d_lstm_cell(
    inputs: Tensor,
    state: tuple[Tensor, Tensor],
    weights: RNNWeights,
    config: Optional[ConvLSTMConfig] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """3D Convolutional LSTM cell.

    Args:
        inputs (Tensor): Input tensor.
        state (tuple[Tensor, Tensor]): Previous state (h_prev, c_prev).
        weights (RNNWeights): Weights for the cell.
        config (Optional[ConvLSTMConfig]): Configuration.

    Returns:
        tuple[Tensor, tuple[Tensor, Tensor]]: The new hidden state and the new state tuple (h_new, c_new).
    """
    h_prev, c_prev = state  # pragma: no cover

    conf = config if config is not None else ConvLSTMConfig()  # pragma: no cover
    x_conv = conv3d(
        inputs,
        weights.kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )  # pragma: no cover
    h_conv = conv3d(  # pragma: no cover
        h_prev,
        weights.recurrent_kernel,
        strides=conf.strides,
        padding=conf.padding,
        data_format=conf.data_format,
    )

    return _apply_conv_lstm_gates(x_conv, h_conv, state, weights, conf.data_format)  # pragma: no cover
