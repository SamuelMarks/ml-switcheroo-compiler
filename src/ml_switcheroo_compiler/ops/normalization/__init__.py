# ruff: noqa
"""Apply normalization operations package."""

from ml_switcheroo_compiler.ops.base import get_op

from .basic import GroupMean, GroupNorm, GroupVariance
from .frontend import NormConfig, group_mean, group_norm, group_variance, spectral_normalization

__all__ = [
    "NormConfig",
    "group_variance",
]
