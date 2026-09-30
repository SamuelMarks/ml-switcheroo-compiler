"""RNN operations frontend."""

from ml_switcheroo_compiler.ops.nn.conv_lstm import conv1d_lstm_cell, conv2d_lstm_cell, conv3d_lstm_cell, conv_lstm_cell
from ml_switcheroo_compiler.ops.nn.gru import gru_cell
from ml_switcheroo_compiler.ops.nn.lstm import lstm_cell
from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    BidirectionalConfig,
    BidirectionalInputs,
    ConvLSTMConfig,
    RNNConfig,
    RNNWeights,
    ScanConfig,
    bidirectional,
    rnn,
    scan,
)

rnn_step = simple_rnn_cell
lstm_step = lstm_cell
gru_step = gru_cell

__all__ = [
    "BidirectionalConfig",
    "BidirectionalInputs",
    "ConvLSTMConfig",
    "RNNConfig",
    "RNNWeights",
    "ScanConfig",
    "bidirectional",
    "conv1d_lstm_cell",
    "conv2d_lstm_cell",
    "conv3d_lstm_cell",
    "conv_lstm_cell",
    "gru_cell",
    "gru_step",
    "lstm_cell",
    "lstm_step",
    "rnn",
    "rnn_step",
    "scan",
    "simple_rnn_cell",
]
