"""Control flow ops."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_eager_list, as_int, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np associative scan operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if len(args) < 2:
        return args[0]
    fn = args[0]
    elems: list[EagerValue] = as_eager_list(args[1])
    axis: int = as_int(kwargs.get("axis", 0))

    elems_arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(elems)
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.empty_like(elems_arr)

    elems_arr = np.moveaxis(elems_arr, axis, 0)
    out = np.moveaxis(out, axis, 0)

    acc: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = as_ndarray(as_callable(fn)(acc, elems_arr[i]))
        out[i] = acc

    out = np.moveaxis(out, 0, axis)
    return out


@numpy_eager_registry.register("StopGradient")
def _stop_gradient(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager stop gradient operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.array(x)
