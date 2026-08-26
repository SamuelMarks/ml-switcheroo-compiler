"""Module dirichlet.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for dirichlet.py."""

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def dirichlet(key, alpha, shape=None, dtype=None):
    """Sample dirichlet random values from a given key.

    Args:
        key (object): The key parameter.
        alpha (object): The alpha parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Dirichlet", [key, alpha], shape, dtype)
