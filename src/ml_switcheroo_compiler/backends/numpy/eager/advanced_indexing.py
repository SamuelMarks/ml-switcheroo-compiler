"""Advanced indexing and scatter/gather operations."""

from __future__ import annotations

import threading
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

from .indexing import IndexingContext


def _gather_nd(x: EagerValue, indices: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): The x parameter.
        indices (EagerValue): The indices parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return typing.cast("EagerValue", getattr(x, "__getitem__")(tuple(getattr(np, "moveaxis")(indices, -1, 0))))


def _scatter_nd(indices: EagerValue, updates: EagerValue, shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.
        shape (EagerValue): The shape parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "zeros")(shape, dtype=getattr(updates, "dtype"))
    res[tuple(getattr(np, "moveaxis")(indices, -1, 0))] = updates
    return res


def _scatter(x: EagerValue, index: EagerValue, src: EagerValue, dim: int, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): The x parameter.
        index (EagerValue): The index parameter.
        src (EagerValue): The src parameter.
        dim (int): The dim parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(x)
    getattr(np, "put_along_axis")(y, index, src, axis=dim)
    return y


def _scatter_add(x: EagerValue, index: EagerValue, src: EagerValue, dim: int, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): The x parameter.
        index (EagerValue): The index parameter.
        src (EagerValue): The src parameter.
        dim (int): The dim parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(x)
    it: np.nditer = getattr(np, "nditer")(index, flags=["multi_index"])
    for idx_val in it:
        pos: list[int] = list(it.multi_index)
        pos[dim] = int(np.asarray(idx_val).item())
        typing.cast(EagerTensorProtocol, y)[tuple(pos)] += getattr(src, "__getitem__")(it.multi_index)
    return y


def _tensor_scatter_update(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter update for numpy.

    Args:
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(getattr(np, "moveaxis")(indices, -1, 0))
    res[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter add for numpy.

    Args:
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(getattr(np, "moveaxis")(indices, -1, 0))
    getattr(np.add, "at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter max for numpy.

    Args:
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(getattr(np, "moveaxis")(indices, -1, 0))
    getattr(np.maximum, "at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter min for numpy.

    Args:
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(getattr(np, "moveaxis")(indices, -1, 0))
    getattr(np.minimum, "at")(res, idx_tuple, updates)
    return res


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: BackendModuleProtocol, x: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
    """Execute the eager np take along axis operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        indices (EagerValue): The indices parameter.
        axis (EagerValue): The axis parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter update operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return typing.cast(TypedVariadicCallable[EagerValue], global_eager_registry.get("TensorScatterUpdate"))(backend_module, tensor, indices, updates)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter add operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter max operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter min operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: BackendModuleProtocol, params: EagerValue, indices: EagerValue) -> EagerValue:
    """Execute the eager np gather nd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        params (EagerValue): The params parameter.
        indices (EagerValue): The indices parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return typing.cast("EagerValue", getattr(params, "__getitem__")(idx))


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(backend_module: BackendModuleProtocol, indices: EagerValue, updates: EagerValue, shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter nd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.
        shape (EagerValue): The shape parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "zeros")(shape, dtype=getattr(updates, "dtype"))
    idx: tuple[EagerValue, ...] = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[0])
    index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[2])
    dim: int = typing.cast(int, kwargs.get("dim", 0))
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.copy(input_data)
    getattr(np, "put_along_axis")(out, index, src, axis=dim)
    return out


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter add operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(args[0])
    index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[2])
    dim: int = typing.cast(int, kwargs.get("dim", 0))
    getattr(np, "put_along_axis")(input_data, index, np.add(getattr(np, "take_along_axis")(input_data, index, axis=dim), src), axis=dim)
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(backend_module: BackendModuleProtocol, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter apply operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    if len(args) == 0:
        return None
    tensor: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0]).copy()
    if len(args) < 3:
        return tensor

    indices: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[1])
    updates: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[2])
    reduction: str | None = typing.cast("str | None", kwargs.get("reduction", None))

    try:
        if reduction == "add":
            np.add.at(tensor, tuple(indices.T), updates)
        elif reduction == "mul":
            np.multiply.at(tensor, tuple(indices.T), updates)
        else:
            tensor[tuple(indices.T)] = updates
    except Exception as e:
        raise RuntimeError(f"Eager execution failed in ScatterNd: {e}") from e
    return tensor


@numpy_eager_registry.register("ScatterMax")
def _np_scatter_max(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue, context: threading.Thread | IndexingContext | None = None) -> EagerValue:
    """Execute the eager np scatter max operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.maximum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue, context: threading.Thread | IndexingContext | None = None) -> EagerValue:
    """Execute the eager np scatter min operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.minimum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue, context: threading.Thread | IndexingContext | None = None) -> EagerValue:
    """Execute the eager np scatter mul operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        tensor (EagerValue): The tensor parameter.
        indices (EagerValue): The indices parameter.
        updates (EagerValue): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.multiply.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor
