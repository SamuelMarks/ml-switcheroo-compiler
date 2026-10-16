"""Apply normalization operations package."""

from ml_switcheroo_compiler.ops.base import get_op  # noqa: F401

from .basic import GroupMean, GroupNorm, GroupVariance  # noqa: F401
from .frontend import NormConfig, group_mean, group_norm, group_variance, spectral_normalization  # noqa: F401

__all__ = [
    "NormConfig",
    "group_variance",
]
