# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module rnn.py."""

"""RNN operations frontend."""

from .conv_lstm import conv1d_lstm_cell, conv2d_lstm_cell, conv3d_lstm_cell, conv_lstm_cell
from .gru import gru_cell
from .lstm import lstm_cell
from .rnn_cell import simple_rnn_cell
from .rnn_utils import (
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
