# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

from typing import Any

"""Apply normalization operations package."""

from ml_switcheroo_compiler.ops.base import get_op

from .basic import GroupMean, GroupNorm, GroupVariance
from .frontend import NormConfig, group_mean, group_norm, group_variance, spectral_normalization

__all__ = [
    "NormConfig",
    "group_variance",
]
