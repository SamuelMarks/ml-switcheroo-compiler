"""Numpy eager scatter/gather operations."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.indexing import _dynamic_update_slice


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(
    backend_module: ModuleType,
    tensor: np.ndarray,
    indices: np.ndarray,
    updates: np.ndarray,
) -> np.ndarray:
    """Evaluate tensor scatter update operation.

    Args:
        backend_module (ModuleType): Active backend module.
        tensor (np.ndarray): Target tensor.
        indices (np.ndarray): Scatter indices.
        updates (np.ndarray): Values to scatter.

    Returns:
        np.ndarray: Updated tensor.
    """
    res = global_eager_registry.get("TensorScatterUpdate")(backend_module, tensor, indices, updates)
    return np.asarray(res)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(
    backend_module: ModuleType,
    tensor: np.ndarray,
    indices: np.ndarray,
    updates: np.ndarray,
) -> np.ndarray:
    """Evaluate tensor scatter add operation.

    Args:
        backend_module (ModuleType): Active backend module.
        tensor (np.ndarray): Target tensor.
        indices (np.ndarray): Scatter indices.
        updates (np.ndarray): Values to add.

    Returns:
        np.ndarray: Tensor with accumulated updates.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(
    backend_module: ModuleType,
    tensor: np.ndarray,
    indices: np.ndarray,
    updates: np.ndarray,
) -> np.ndarray:
    """Evaluate tensor scatter maximum operation.

    Args:
        backend_module (ModuleType): Active backend module.
        tensor (np.ndarray): Target tensor.
        indices (np.ndarray): Scatter indices.
        updates (np.ndarray): Values to take maximum with.

    Returns:
        np.ndarray: Tensor with maximum applied.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(
    backend_module: ModuleType,
    tensor: np.ndarray,
    indices: np.ndarray,
    updates: np.ndarray,
) -> np.ndarray:
    """Evaluate tensor scatter minimum operation.

    Args:
        backend_module (ModuleType): Active backend module.
        tensor (np.ndarray): Target tensor.
        indices (np.ndarray): Scatter indices.
        updates (np.ndarray): Values to take minimum with.

    Returns:
        np.ndarray: Tensor with minimum applied.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: ModuleType,
    indices: Union[np.ndarray, Sequence[int]],
    updates: np.ndarray,
    shape: tuple[int, ...],
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Evaluate scatter_nd operation creating zero tensor with scattered updates.

    Args:
        backend_module (ModuleType): Active backend module.
        indices (Union[np.ndarray, Sequence[int]]): Index coordinates.
        updates (np.ndarray): Source updates tensor.
        shape (tuple[int, ...]): Output shape tuple.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Scattered tensor of specified shape.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int],
    **kwargs: Union[int, None],
) -> np.ndarray:
    """Evaluate scatter operation along a dimension.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int]): Positional arguments (input_data, index, src).
        **kwargs (Union[int, None]): Keyword arguments such as dim.

    Returns:
        np.ndarray: Scattered array along specified axis.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim_val = kwargs.get("dim", 0)
    dim = int(dim_val) if dim_val is not None else 0
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


def _band_part(
    input_tensor: np.ndarray,
    num_lower: int,
    num_upper: int,
) -> np.ndarray:
    """Evaluate band_part matrix operation.

    Args:
        input_tensor (np.ndarray): Input tensor of matrices.
        num_lower (int): Number of subdiagonals to keep.
        num_upper (int): Number of superdiagonals to keep.

    Returns:
        np.ndarray: Band-part extracted matrix tensor.
    """
    input_arr = np.asarray(input_tensor)
    res = np.copy(input_arr)
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(
    backend_module: ModuleType,
    params: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Evaluate gather_nd operation.

    Args:
        backend_module (ModuleType): Active backend module.
        params (np.ndarray): Source parameters tensor.
        indices (np.ndarray): Indices slice coordinates.

    Returns:
        np.ndarray: Gathered tensor elements.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(
    backend_module: ModuleType,
    x: np.ndarray,
    indices: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Evaluate take_along_axis operation.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Source values array.
        indices (np.ndarray): 1D or ND indices array.
        axis (int): Axis along which to take.

    Returns:
        np.ndarray: Selected slice values.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(
    backend_module: ModuleType,
    x: np.ndarray,
    start_indices: Sequence[int],
    slice_sizes: Sequence[int],
) -> np.ndarray:
    """Evaluate dynamic slicing from variable starts with fixed sizes.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Source tensor.
        start_indices (Sequence[int]): Starting coordinates per axis.
        slice_sizes (Sequence[int]): Size of slice per axis.

    Returns:
        np.ndarray: Dynamically sliced subarray.
    """
    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(
    backend_module: ModuleType,
    *args: Union[np.ndarray, Sequence[int]],
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Evaluate dynamic update slice operation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, Sequence[int]]): Positional arguments (operand, update, start_indices).
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Updated array with slice replaced.
    """
    return _dynamic_update_slice(*args, **kwargs)
