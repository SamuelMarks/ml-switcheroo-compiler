# ruff: noqa
# auto-generate-all
"""Neural network operations and layers for the ML Switcheroo compiler."""

from ml_switcheroo_compiler.ops.reductions import adaptive_avg_pool2d, adaptive_max_pool2d
from .rnn_utils import rnn as rnn
from .rnn_utils import RNNConfig as RNNConfig
from .pooling import max_pool as max_pool
from .pooling import avg_pool as avg_pool
from .pooling import max_pool1d as max_pool1d
from .pooling import max_pool2d as max_pool2d
from .pooling import max_pool3d as max_pool3d
from .pooling import avg_pool1d as avg_pool1d
from .pooling import avg_pool2d as avg_pool2d
from .pooling import avg_pool3d as avg_pool3d
from .lstm import lstm_cell as lstm_cell
from .gru import gru_cell as gru_cell
from .gru import gru as gru
from .conv_lstm import conv1d_lstm_cell as conv1d_lstm_cell
from .conv_lstm import conv2d_lstm_cell as conv2d_lstm_cell
from .conv_lstm import conv3d_lstm_cell as conv3d_lstm_cell
