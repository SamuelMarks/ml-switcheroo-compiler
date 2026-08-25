"""Module truncated_normal.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for truncated_normal.py."""

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def truncated_normal(key: object, lower: object, upper: object, shape: object = (), dtype: object = None) -> object:
    """Return an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (object): The key parameter.
        lower (object): The lower parameter.
        upper (object): The upper parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: object = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper})
