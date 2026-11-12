"""Advanced indexing and scatter/gather operations."""

from __future__ import annotations

import threading

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, ensure_callable, ensure_eager_tensor, ensure_int, ensure_opt_str, global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member

from .indexing import IndexingContext


def _gather_nd(x: EagerValue, indices: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): Input `x` argument (x).
        indices (EagerValue): Input `indices` argument (indices).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(ensure_eager_tensor(x)[tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))])


def _scatter_nd(indices: EagerValue, updates: EagerValue, shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).
        shape (EagerValue): Input `shape` argument (shape).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "zeros")(shape, dtype=get_backend_member(updates, "dtype")))
    res[tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))] = ensure_ndarray(updates)
    return res


def _scatter(x: EagerValue, index: EagerValue, src: EagerValue, dim: int, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): Input `x` argument (x).
        index (EagerValue): Input `index` argument (index).
        src (EagerValue): Input `src` argument (src).
        dim (int): Input `dim` argument (dim).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(x))
    get_backend_member(np, "put_along_axis")(y, index, src, axis=dim)
    return y


def _scatter_add(x: EagerValue, index: EagerValue, src: EagerValue, dim: int, **kwargs: EagerValue) -> EagerValue:
    """Evaluate.

    Args:
        x (EagerValue): Input `x` argument (x).
        index (EagerValue): Input `index` argument (index).
        src (EagerValue): Input `src` argument (src).
        dim (int): Input `dim` argument (dim).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(x))
    it = np.nditer(np.asarray(index), flags=["multi_index"])
    for idx_val in it:
        pos: list[int] = list(it.multi_index)
        pos[dim] = int(np.asarray(idx_val).item())
        y[tuple(pos)] += ensure_eager_tensor(src)[it.multi_index]
    return y


def _tensor_scatter_update(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter update for numpy.

    Args:
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(tensor))
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))
    ensure_eager_tensor(res)[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter add for numpy.

    Args:
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(tensor))
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))
    get_backend_member(np.add, "at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter max for numpy.

    Args:
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(tensor))
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))
    get_backend_member(np.maximum, "at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Tensor scatter min for numpy.

    Args:
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(tensor))
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[EagerValue, ...] = tuple(get_backend_member(np, "moveaxis")(indices, -1, 0))
    get_backend_member(np.minimum, "at")(res, idx_tuple, updates)
    return res


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: BackendModuleProtocol, x: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
    """Execute the eager np take along axis operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        indices (EagerValue): Input `indices` argument (indices).
        axis (EagerValue): Input `axis` argument (axis).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter update operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(global_eager_registry.get("TensorScatterUpdate"))(backend_module, tensor, indices, updates)


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter add operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter max operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue) -> EagerValue:
    """Execute the eager np tensor scatter min operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: BackendModuleProtocol, params: EagerValue, indices: EagerValue) -> EagerValue:
    """Execute the eager np gather nd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        params (EagerValue): Input `params` argument (params).
        indices (EagerValue): Input `indices` argument (indices).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    idx: tuple[EagerValue, ...] = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return ensure_eager_tensor(ensure_eager_tensor(params)[idx])


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(backend_module: BackendModuleProtocol, indices: EagerValue, updates: EagerValue, shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter nd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).
        shape (EagerValue): Input `shape` argument (shape).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "zeros")(shape, dtype=get_backend_member(updates, "dtype")))
    idx: tuple[EagerValue, ...] = tuple(np.moveaxis(np.array(indices), -1, 0))
    ensure_eager_tensor(out)[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[0])
    index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[2])
    dim: int = ensure_int(kwargs.get("dim", 0))
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.copy(input_data)
    get_backend_member(np, "put_along_axis")(out, index, src, axis=dim)
    return out


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter add operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "copy")(args[0]))
    index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[2])
    dim: int = ensure_int(kwargs.get("dim", 0))
    get_backend_member(np, "put_along_axis")(input_data, index, np.add(ensure_ndarray(get_backend_member(np, "take_along_axis")(input_data, index, axis=dim)), src), axis=dim)
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(backend_module: BackendModuleProtocol, context: IndexingContext | EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatter apply operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        context (IndexingContext): Input `context` argument (context).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    if len(args) == 0:
        return None
    tensor: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(args[0]).copy()
    if len(args) < 3:
        return tensor

    indices: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(args[1])
    updates: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(args[2])
    reduction: str | None = ensure_opt_str(kwargs.get("reduction", None))

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
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).
        context (IndexingContext): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.maximum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue, context: threading.Thread | IndexingContext | None = None) -> EagerValue:
    """Execute the eager np scatter min operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).
        context (IndexingContext): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.minimum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(backend_module: BackendModuleProtocol, tensor: EagerValue, indices: EagerValue, updates: EagerValue, context: threading.Thread | IndexingContext | None = None) -> EagerValue:
    """Execute the eager np scatter mul operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        tensor (EagerValue): Input `tensor` argument (tensor).
        indices (EagerValue): Input `indices` argument (indices).
        updates (EagerValue): Input `updates` argument (updates).
        context (IndexingContext): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.multiply.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor
