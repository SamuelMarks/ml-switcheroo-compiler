# ruff: noqa: E501
"""Numpy eager scatter/gather operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .indexing import _dynamic_update_slice


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate the tensor scatter update logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.

    Returns:
        object: The evaluated or processed output.
    """
    return global_eager_registry.get("TensorScatterUpdate")(backend_module, tensor, indices, updates)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate the tensor scatter add logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.

    Returns:
        object: The evaluated or processed output.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate the tensor scatter max logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.

    Returns:
        object: The evaluated or processed output.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: object, tensor: object, indices: object, updates: object) -> object:
    """Evaluate the tensor scatter min logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.

    Returns:
        object: The evaluated or processed output.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(backend_module: object, indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate the scatter nd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.
        shape (object): Required parameter for shape.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the scatter logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    """Evaluate and process the band part operation.

    Args:
        input (object): Required parameter for input.
        num_lower (object): Required parameter for num_lower.
        num_upper (object): Required parameter for num_upper.

    Returns:
        object: The evaluated or processed output.
    """
    input = np.asarray(input)
    (m, n) = input.shape[-2:]
    res = np.copy(input)
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    """Evaluate the gather nd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        params (object): Required parameter for params.
        indices (object): Required parameter for indices.

    Returns:
        object: The evaluated or processed output.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    """Evaluate the take along axis logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        indices (object): Required parameter for indices.
        axis (object): Required parameter for axis.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: object, x: object, start_indices: object, slice_sizes: object) -> object:
    """Evaluate the dynamic slice logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        start_indices (object): Required parameter for start_indices.
        slice_sizes (object): Required parameter for slice_sizes.

    Returns:
        object: The evaluated or processed output.
    """
    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the dynamic update slice logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _dynamic_update_slice(*args, **kwargs)
