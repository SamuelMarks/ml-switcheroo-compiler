from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for uniform.py."""
from typing import Any

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def uniform(key: Any, shape: Any = (), dtype: Any = None, minval: Any = 0.0, maxval: Any = 1.0) -> Any:
    """Sample uniform random values from a given key.

    Args:
        key (object): The key parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.
        minval (object): The minval parameter.
        maxval (object): The maxval parameter.

    Returns: Any: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomUniform", [key], shape, dtype, {"minval": minval, "maxval": maxval})
