# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Control flow ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module: object, *args: object, **kwargs: object) -> object:
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
    fn: object = args[0]
    elems: object = args[1]
    axis: object = kwargs.get("axis", 0)

    elems_arr: object = np.asarray(elems)
    out: object = np.empty_like(elems_arr)

    elems_arr: object = np.moveaxis(elems_arr, axis, 0)
    out: object = np.moveaxis(out, axis, 0)

    acc: object = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc: object = fn(acc, elems_arr[i])
        out[i] = acc

    out: object = np.moveaxis(out, 0, axis)
    return out


@numpy_eager_registry.register("StopGradient")
def _stop_gradient(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
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
