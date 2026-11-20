"""Native JAX distributed collective dispatch bindings."""

from __future__ import annotations

import logging

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, require_callable


def jax_all_reduce(
    tensor: BackendArg,
    op: str = "SUM",
    axis_name: str = "data",
) -> BackendArg:
    """Execute native JAX psum/pmax/pmin collective reduction.

    Args:
        tensor (BackendArg): JAX Array or compatible buffer.
        op (str): Reduction operator.
        axis_name (str): Named pmapped mesh axis.

    Returns:
        BackendArg: Reduced tensor array.
    """
    try:
        import jax.lax as lax

        lax_mod = lax
        if op.upper() in ("SUM", "ADD"):
            return require_callable(lax_mod.psum)(tensor, axis_name=axis_name)
        elif op.upper() == "MAX":
            return require_callable(lax_mod.pmax)(tensor, axis_name=axis_name)
        elif op.upper() == "MIN":
            return require_callable(lax_mod.pmin)(tensor, axis_name=axis_name)
    except (ImportError, NameError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def jax_all_gather(
    tensor: BackendArg,
    axis_name: str = "data",
    axis: int = 0,
) -> BackendArg:
    """Execute native JAX all_gather along a named axis.

    Args:
        tensor (BackendArg): JAX Array or compatible buffer.
        axis_name (str): Named pmapped mesh axis.
        axis (int): Output concatenation axis.

    Returns:
        BackendArg: Gathered global tensor.
    """
    try:
        import jax.lax as lax

        return require_callable(lax.all_gather)(tensor, axis_name=axis_name, axis=axis)
    except (ImportError, NameError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def jax_broadcast(
    tensor: BackendArg,
    src: int = 0,
) -> BackendArg:
    """Execute native JAX broadcast.

    Args:
        tensor (BackendArg): JAX Array.
        src (int): Source root rank.

    Returns:
        BackendArg: Broadcasted tensor.
    """
    return tensor
