# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module conv.py."""

"""Convolution operations frontend."""

from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated as conv_general

from .conv1d import conv1d, conv1d_transpose, depthwise_conv1d, separable_conv1d
from .conv2d import conv2d, conv2d_transpose, depthwise_conv2d, separable_conv2d
from .conv3d import conv3d, conv3d_transpose
from .conv_nd import conv, conv_transpose, depthwise_conv, separable_conv
from .conv_utils import GenericConvConfig

conv_transpose1d: object = conv1d_transpose
conv_transpose2d: object = conv2d_transpose
conv_transpose3d: object = conv3d_transpose

__all__ = [
    "GenericConvConfig",
    "conv",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "conv3d_transpose",
    "conv_general",
    "conv_transpose",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "depthwise_conv",
    "depthwise_conv1d",
    "depthwise_conv2d",
    "separable_conv",
    "separable_conv1d",
    "separable_conv2d",
]
