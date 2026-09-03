# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_general import _get_np_arg, _get_sc


@numpy_eager_registry.register("Pmean")
def _np_pmean(backend_module, x, axis_name, *args, **kwargs):
    """Evaluate _np_pmean operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis_name (object): The axis_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


@numpy_eager_registry.register("Psum")
def _np_psum(backend_module, *args, **kwargs):
    """Evaluate _np_psum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    if _tcp_dist_ctx.world_size > 1:
        tensor = backend_module.array(args[0])
        return _tcp_dist_ctx.all_reduce_ring(tensor, op_type="sum", backend_module=backend_module)
    return backend_module.array(args[0])


@numpy_eager_registry.register("ReducePrecision")
def _np_reduce_precision(backend_module, x, exponent_bits: int, mantissa_bits: int):
    """Reduce the precision of a tensor to a specified number of exponent and mantissa bits.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        exponent_bits (int): The exponent_bits parameter.
        mantissa_bits (int): The mantissa_bits parameter.

    Returns: np.ndarray: The computed result.
    """
    return np.asarray(x).astype(np.float16).astype(np.asarray(x).dtype)


@numpy_eager_registry.register("SparseReduceMax")
def _np_sparsereducemax(backend_module, *args, **kwargs):
    """Implement SparseReduceMax.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.max(args[0], axis=-1)
