# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module tensor.py."""

"""Tensor Manipulation operations."""

from ml_switcheroo_compiler.ops.binary import allclose
from ml_switcheroo_compiler.ops.shape import take_along_axis

__all__ = [
    "allclose",
    "take_along_axis",
]
