# ruff: noqa: E501
"""Control flow ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the associative scan logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (object): Required parameter for elems.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    if len(args) < 2:
        return args[0]
    fn = args[0]
    elems = args[1]
    axis = kwargs.get("axis", 0)

    elems_arr = np.asarray(elems)
    out = np.empty_like(elems_arr)

    elems_arr = np.moveaxis(elems_arr, axis, 0)
    out = np.moveaxis(out, axis, 0)

    acc = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = fn(acc, elems_arr[i])
        out[i] = acc

    out = np.moveaxis(out, 0, axis)
    return out


@numpy_eager_registry.register("StopGradient")
def _stop_gradient(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    return backend_module.array(x)
