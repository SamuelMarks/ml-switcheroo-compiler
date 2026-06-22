"""Normalization operations package."""

import ml_switcheroo_compiler.ops.normalization.basic as _basic
from ml_switcheroo_compiler.ops.base import get_op

from .frontend import NormConfig, group_mean, group_norm, group_variance, spectral_normalization

_ = _basic

__all__ = [
    "group_mean",
    "group_variance",
    "group_norm",
    "spectral_normalization",
    "NormConfig",
    "get_op",
]
