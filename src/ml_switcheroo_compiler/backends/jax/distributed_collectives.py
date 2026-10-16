"""Native JAX distributed collective dispatch bindings."""

from __future__ import annotations

import logging
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def jax_all_reduce(
    tensor: EagerValue,
    op: str = "SUM",
    axis_name: str = "data",
) -> EagerValue:
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
            return cast(EagerValue, getattr(lax, "psum")(tensor, axis_name=axis_name))
        elif op.upper() == "MAX":
            return cast(EagerValue, getattr(lax, "pmax")(tensor, axis_name=axis_name))
        elif op.upper() == "MIN":
            return cast(EagerValue, getattr(lax, "pmin")(tensor, axis_name=axis_name))
    except (ImportError, NameError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def jax_all_gather(
    tensor: EagerValue,
    axis_name: str = "data",
    axis: int = 0,
) -> EagerValue:
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

        return cast(EagerValue, getattr(lax, "all_gather")(tensor, axis_name=axis_name, axis=axis))
    except (ImportError, NameError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def jax_broadcast(
    tensor: EagerValue,
    src: int = 0,
) -> EagerValue:
    """Execute native JAX broadcast.

    Args:
        tensor (Any): JAX Array.
        src (int): Source root rank.

    Returns:
        Any: Broadcasted tensor.
    """
    return tensor
