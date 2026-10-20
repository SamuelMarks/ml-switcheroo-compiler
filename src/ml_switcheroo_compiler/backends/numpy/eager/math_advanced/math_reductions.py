"""Math Ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

ArrayInput = EagerValue


@numpy_eager_registry.register("Pmean")
def _np_pmean(
    backend_module: BackendModuleProtocol,
    x: ArrayInput,
    axis_name: str,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> ArrayInput:
    """Execute the eager np pmean operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        x (ArrayInput): Input array.
        axis_name (str): Named axis.
        *args (Union[float, int, str, bool, None]): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        ArrayInput: Result.
    """
    return x


@numpy_eager_registry.register("Psum")
def _np_psum(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np psum operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    if _tcp_dist_ctx.world_size > 1:
        tensor = np.asarray(backend_module.array(args[0]))
        return _tcp_dist_ctx.all_reduce_ring(tensor, op_type="sum", backend_module=backend_module)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(args[0]))


@numpy_eager_registry.register("ReducePrecision")
def _np_reduce_precision(
    backend_module: BackendModuleProtocol,
    x: ArrayInput,
    exponent_bits: int,
    mantissa_bits: int,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Reduce the precision of a tensor to a specified number of exponent and mantissa bits.

    Args:
        backend_module (ModuleType): The backend module.
        x (ArrayInput): Input array.
        exponent_bits (int): The exponent bits count.
        mantissa_bits (int): The mantissa bits count.

    Returns:
        np.ndarray: The computed result.
    """
    return np.asarray(x).astype(np.float16).astype(np.asarray(x).dtype)


@numpy_eager_registry.register("SparseReduceMax")
def _np_sparsereducemax(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement SparseReduceMax.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.max(args[0], axis=-1))
