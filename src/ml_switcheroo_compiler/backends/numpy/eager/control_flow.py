# ruff: noqa: E402, E501, F403
"""Control flow ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module, *args, **kwargs):
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
    fn: np.ndarray = args[0]
    elems: list = args[1]
    axis: int = kwargs.get("axis", 0)

    elems_arr: np.ndarray = np.asarray(elems)
    out: np.ndarray = np.empty_like(elems_arr)

    elems_arr: np.ndarray = np.moveaxis(elems_arr, axis, 0)
    out: np.ndarray = np.moveaxis(out, axis, 0)

    acc: np.ndarray = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc: np.ndarray = fn(acc, elems_arr[i])
        out[i] = acc

    out: np.ndarray = np.moveaxis(out, 0, axis)
    return out


@numpy_eager_registry.register("StopGradient")
def _stop_gradient(backend_module, x, *args, **kwargs):
    """Evaluate _stop_gradient operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.array(x)
