"""Normalization operations package."""

from ml_switcheroo_compiler.ops.base import get_op
import ml_switcheroo_compiler.ops.normalization.basic as _basic
from .frontend import group_mean, group_variance, group_norm

_ = _basic

__all__ = ["group_mean", "group_variance", "group_norm", "get_op"]
