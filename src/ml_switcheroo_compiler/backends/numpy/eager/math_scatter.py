# ruff: noqa: E501
"""Numpy eager scatter/gather operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .indexing import _dynamic_update_slice


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate _np_tensor_scatter_update operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    return global_eager_registry.get("TensorScatterUpdate")(backend_module, tensor, indices, updates)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate _np_tensor_scatter_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate _np_tensor_scatter_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate _np_tensor_scatter_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(backend_module: object, indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate _np_scatter_nd operation.

    Args:
        backend_module (object): The backend_module parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_scatter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    """Evaluate _band_part operation.

    Args:
        input (object): The input parameter.
        num_lower (object): The num_lower parameter.
        num_upper (object): The num_upper parameter.

    Returns:
        object: Result.
    """
    input = np.asarray(input)
    (m, n) = input.shape[-2:]
    res = np.copy(input)
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    """Evaluate _np_gather_nd operation.

    Args:
        backend_module (object): The backend_module parameter.
        params (object): The params parameter.
        indices (object): The indices parameter.

    Returns:
        object: Result.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    """Evaluate _np_take_along_axis operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        indices (object): The indices parameter.
        axis (object): The axis parameter.

    Returns:
        object: Result.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: object, x: object, start_indices: object, slice_sizes: object) -> object:
    """Evaluate _np_dynamic_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        start_indices (object): The start_indices parameter.
        slice_sizes (object): The slice_sizes parameter.

    Returns:
        object: Result.
    """
    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_update_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dynamic_update_slice(*args, **kwargs)
