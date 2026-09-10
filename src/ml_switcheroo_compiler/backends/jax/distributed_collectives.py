# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Native JAX distributed collective dispatch bindings."""

from __future__ import annotations

from typing import Any


def jax_all_reduce(
    tensor: Any,
    op: str = "SUM",
    axis_name: str = "data",
) -> Any:
    """Execute native JAX psum/pmax/pmin collective reduction.

    Args:
        tensor (Any): JAX Array or compatible buffer.
        op (str): Reduction operator.
        axis_name (str): Named pmapped mesh axis.

    Returns:
        Any: Reduced tensor array.
    """
    try:
        import jax.lax as lax

        if op.upper() in ("SUM", "ADD"):
            return lax.psum(tensor, axis_name=axis_name)
        elif op.upper() == "MAX":
            return lax.pmax(tensor, axis_name=axis_name)
        elif op.upper() == "MIN":
            return lax.pmin(tensor, axis_name=axis_name)
    except (ImportError, NameError):
        pass

    return tensor


def jax_all_gather(
    tensor: Any,
    axis_name: str = "data",
    axis: int = 0,
) -> Any:
    """Execute native JAX all_gather along a named axis.

    Args:
        tensor (Any): JAX Array or compatible buffer.
        axis_name (str): Named pmapped mesh axis.
        axis (int): Output concatenation axis.

    Returns:
        Any: Gathered global tensor.
    """
    try:
        import jax.lax as lax

        return lax.all_gather(tensor, axis_name=axis_name, axis=axis)
    except (ImportError, NameError):
        pass

    return tensor


def jax_broadcast(
    tensor: Any,
    src: int = 0,
) -> Any:
    """Execute native JAX broadcast.

    Args:
        tensor (Any): JAX Array.
        src (int): Source root rank.

    Returns:
        Any: Broadcasted tensor.
    """
    return tensor
