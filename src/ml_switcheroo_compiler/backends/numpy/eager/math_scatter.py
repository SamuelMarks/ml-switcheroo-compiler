"""Numpy eager scatter/gather operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .shape import _dynamic_update_slice


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    return global_eager_registry.get("TensorScatterUpdate")(backend_module, tensor, indices, updates)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: object,
    indices: object,
    updates: object,
    shape: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        indices: Arg.
        updates: Arg.
        shape: Arg.
        kwargs: Arg.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


# Import op groups to register them


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    """Function docstring."""
    input = np.asarray(input)
    m, n = input.shape[-2:]
    res = np.copy(input)
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        params: Arg.
        indices: Arg.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        indices: Arg.
        axis: Arg.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: object, x: object, start_indices: object, slice_sizes: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        start_indices: Arg.
        slice_sizes: Arg.
    """
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _dynamic_update_slice(*args, **kwargs)
