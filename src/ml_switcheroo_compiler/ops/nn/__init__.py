"""Neural network operations."""

from .activations import elu, gelu, relu, selu, softplus
from .conv import (
    conv1d,
    conv1d_transpose,
    conv2d,
    conv2d_transpose,
    conv3d,
    conv3d_transpose,
    conv_transpose,
    depthwise_conv1d,
    depthwise_conv2d,
    separable_conv1d,
    separable_conv2d,
)
from .dropout import dropout, activity_regularization
from .loss import ctc_loss, dice_loss, categorical_generalized_cross_entropy, circle_loss
from .normalization import local_response_normalization
from .nlp import AttentionConfig, AttentionInputs, attention, embedding
from .time_distributed import time_distributed
from .pooling import avg_pool, max_pool, pool1d, pool2d, pool3d
from .rnn import (
    gru_cell,
    lstm_cell,
    simple_rnn_cell,
    rnn,
    scan,
    conv_lstm_cell,
    conv1d_lstm_cell,
    conv2d_lstm_cell,
    conv3d_lstm_cell,
    bidirectional,
)

__all__ = [
    "activity_regularization",
    "attention",
    "avg_pool",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "conv3d_transpose",
    "conv_transpose",
    "depthwise_conv1d",
    "depthwise_conv2d",
    "separable_conv1d",
    "separable_conv2d",
    "dropout",
    "max_pool",
    "pool1d",
    "pool2d",
    "pool3d",
    "rnn",
    "lstm_cell",
    "simple_rnn_cell",
    "gru_cell",
    "conv_lstm_cell",
    "conv1d_lstm_cell",
    "conv2d_lstm_cell",
    "conv3d_lstm_cell",
    "bidirectional",
    "time_distributed",
    "scan",
    "softplus",
    "relu",
    "selu",
    "elu",
    "gelu",
    "ctc_loss",
    "dice_loss",
    "categorical_generalized_cross_entropy",
    "circle_loss",
    "local_response_normalization",
    "embedding",
    "activity_regularization",
    "attention",
    "AttentionInputs",
    "AttentionConfig",
]
