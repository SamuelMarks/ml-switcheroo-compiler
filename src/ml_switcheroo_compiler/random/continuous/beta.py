"""Module beta.py."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for beta.py."""

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def beta(key: Any, a: Any, b: Any, shape: Sequence[int] | int | None = None, dtype: Any | None = None) -> Tensor:
    """Sample beta random values from a given key.

    Args:
        key (object): The key parameter.
        a (object): The a parameter.
        b (object): The b parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Beta", [key, a, b], shape, dtype)
