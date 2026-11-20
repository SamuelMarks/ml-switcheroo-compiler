"""Neural network operations and layers for the ML Switcheroo compiler."""

from __future__ import annotations

# auto-generate-all
from ml_switcheroo_compiler.ops.reductions import adaptive_avg_pool2d, adaptive_max_pool2d  # noqa: F401

from .attention import scaled_dot_product_attention as scaled_dot_product_attention
from .conv import (
    GenericConvConfig as GenericConvConfig,
)
from .conv import (
    conv1d as conv1d,
)
from .conv import (
    conv1d_transpose as conv1d_transpose,
)
from .conv import (
    conv2d as conv2d,
)
from .conv import (
    conv2d_transpose as conv2d_transpose,
)
from .conv import (
    conv3d as conv3d,
)
from .conv import (
    conv3d_transpose as conv3d_transpose,
)
from .conv import (
    conv_general as conv_general,
)
from .conv import (
    conv_transpose as conv_transpose,
)
from .conv import (
    conv_transpose1d as conv_transpose1d,
)
from .conv import (
    conv_transpose2d as conv_transpose2d,
)
from .conv import (
    conv_transpose3d as conv_transpose3d,
)
from .conv import (
    depthwise_conv as depthwise_conv,
)
from .conv import (
    depthwise_conv1d as depthwise_conv1d,
)
from .conv import (
    depthwise_conv2d as depthwise_conv2d,
)
from .conv import (
    separable_conv as separable_conv,
)
from .conv import (
    separable_conv1d as separable_conv1d,
)
from .conv import (
    separable_conv2d as separable_conv2d,
)
from .conv_lstm import conv1d_lstm_cell as conv1d_lstm_cell
from .conv_lstm import conv2d_lstm_cell as conv2d_lstm_cell
from .conv_lstm import conv3d_lstm_cell as conv3d_lstm_cell
from .gru import gru as gru
from .gru import gru_cell as gru_cell
from .loss import (
    AdaptiveLogSoftmaxWithLoss as AdaptiveLogSoftmaxWithLoss,
)
from .loss import (
    CategoricalGeneralizedCrossEntropy as CategoricalGeneralizedCrossEntropy,
)
from .loss import (
    CircleLoss as CircleLoss,
)
from .loss import (
    CtcLoss as CtcLoss,
)
from .loss import (
    InTopK as InTopK,
)
from .loss import (
    LogPoissonLoss as LogPoissonLoss,
)
from .loss import (
    adaptive_log_softmax_with_loss as adaptive_log_softmax_with_loss,
)
from .loss import (
    binary_crossentropy as binary_crossentropy,
)
from .loss import (
    categorical_crossentropy as categorical_crossentropy,
)
from .loss import (
    categorical_generalized_cross_entropy as categorical_generalized_cross_entropy,
)
from .loss import (
    circle_loss as circle_loss,
)
from .loss import (
    cosine_similarity_loss as cosine_similarity_loss,
)
from .loss import (
    ctc_decode as ctc_decode,
)
from .loss import (
    ctc_loss as ctc_loss,
)
from .loss import (
    dice_loss as dice_loss,
)
from .loss import (
    gaussian_nll_loss as gaussian_nll_loss,
)
from .loss import (
    hinge_loss as hinge_loss,
)
from .loss import (
    huber_loss as huber_loss,
)
from .loss import (
    in_top_k as in_top_k,
)
from .loss import (
    kl_div_loss as kl_div_loss,
)
from .loss import (
    kld as kld,
)
from .loss import (
    kullback_leibler_divergence as kullback_leibler_divergence,
)
from .loss import (
    l1_loss as l1_loss,
)
from .loss import (
    l2_loss as l2_loss,
)
from .loss import (
    log_cosh_loss as log_cosh_loss,
)
from .loss import (
    log_poisson_loss as log_poisson_loss,
)
from .loss import (
    logcosh as logcosh,
)
from .loss import (
    mape as mape,
)
from .loss import (
    mape_loss as mape_loss,
)
from .loss import (
    margin_ranking_loss as margin_ranking_loss,
)
from .loss import (
    mse_loss as mse_loss,
)
from .loss import (
    msle as msle,
)
from .loss import (
    msle_loss as msle_loss,
)
from .loss import (
    nll_loss as nll_loss,
)
from .loss import (
    scale_regularization_loss as scale_regularization_loss,
)
from .loss import (
    smooth_l1_loss as smooth_l1_loss,
)
from .loss import (
    sparse_categorical_crossentropy as sparse_categorical_crossentropy,
)
from .loss import (
    triplet_loss as triplet_loss,
)
from .loss import (
    tversky_loss as tversky_loss,
)
from .lstm import lstm_cell as lstm_cell
from .normalization import Normalization as Normalization
from .pooling import avg_pool as avg_pool
from .pooling import avg_pool1d as avg_pool1d
from .pooling import avg_pool2d as avg_pool2d
from .pooling import avg_pool3d as avg_pool3d
from .pooling import max_pool as max_pool
from .pooling import max_pool1d as max_pool1d
from .pooling import max_pool2d as max_pool2d
from .pooling import max_pool3d as max_pool3d
from .rnn_utils import RNNConfig as RNNConfig
from .rnn_utils import rnn as rnn
