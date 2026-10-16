"""Control flow ops."""

from __future__ import annotations

from types import ModuleType
from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_associative_scan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if len(args) < 2:
        return args[0]
    fn = args[0]
    elems: list[EagerValue] = cast(list[EagerValue], args[1])
    axis: int = cast(int, kwargs.get("axis", 0))

    elems_arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(elems)
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.empty_like(elems_arr)

    elems_arr = np.moveaxis(elems_arr, axis, 0)
    out = np.moveaxis(out, axis, 0)

    acc: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = cast("Callable[..., np.ndarray[tuple[int, ...], np.dtype[np.generic]]]", fn)(acc, elems_arr[i])
        out[i] = acc

    out = np.moveaxis(out, 0, axis)
    return out


@numpy_eager_registry.register("StopGradient")
def _stop_gradient(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _stop_gradient operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.array(x))
