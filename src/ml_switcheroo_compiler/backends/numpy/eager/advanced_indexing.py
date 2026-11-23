"""Advanced indexing and scatter/gather operations."""

from __future__ import annotations

import threading
import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_array,
    coerce_int,
    coerce_opt_str,
    global_eager_registry,
    numpy_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

from .indexing import IndexingContext


def _gather_nd(x: BackendArg, indices: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Evaluate.

    Args:
        x (BackendArg): Input `x` argument (x).
        indices (BackendArg): Input `indices` argument (indices).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    idx_tuple: tuple[EagerTensorProtocol, ...] = tuple(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0))
    return coerce_array(np, x)[idx_tuple]


def _scatter_nd(indices: BackendArg, updates: BackendArg, shape: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Evaluate.

    Args:
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).
        shape (BackendArg): Input `shape` argument (shape).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("zeros")(shape, dtype=adapt_backend_module(updates).require_symbol("dtype"))
    )
    res[tuple(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0))] = ensure_ndarray(updates)
    return res


def _scatter(x: BackendArg, index: BackendArg, src: BackendArg, dim: int, **kwargs: BackendArg) -> BackendArg:
    """Evaluate.

    Args:
        x (BackendArg): Input `x` argument (x).
        index (BackendArg): Input `index` argument (index).
        src (BackendArg): Input `src` argument (src).
        dim (int): Input `dim` argument (dim).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(x)
    )
    adapt_backend_module(np).require_op("put_along_axis")(y, index, src, axis=dim)
    return y


def _scatter_add(x: BackendArg, index: BackendArg, src: BackendArg, dim: int, **kwargs: BackendArg) -> BackendArg:
    """Evaluate.

    Args:
        x (BackendArg): Input `x` argument (x).
        index (BackendArg): Input `index` argument (index).
        src (BackendArg): Input `src` argument (src).
        dim (int): Input `dim` argument (dim).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(x)
    )
    src_arr = ensure_ndarray(src)
    it = np.nditer(np.asarray(index), flags=["multi_index"])
    for idx_val in it:
        pos: list[int] = list(it.multi_index)
        pos[dim] = int(np.asarray(idx_val).item())
        y[tuple(pos)] += src_arr[it.multi_index]
    return y


def _tensor_scatter_update(tensor: BackendArg, indices: BackendArg, updates: BackendArg) -> BackendArg:
    """Tensor scatter update for numpy.

    Args:
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(tensor)
    )
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    moved_idx = ensure_ndarray(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0)).astype(np.intp)
    idx_tuple: tuple[np.ndarray[tuple[int, ...], np.dtype[np.intp]], ...] = tuple(
        np.asarray(moved_idx[i], dtype=np.intp) for i in range(moved_idx.shape[0])
    )
    res[idx_tuple] = ensure_ndarray(updates)
    return res


def _tensor_scatter_add(tensor: BackendArg, indices: BackendArg, updates: BackendArg) -> BackendArg:
    """Tensor scatter add for numpy.

    Args:
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(tensor)
    )
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[BackendArg, ...] = tuple(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0))
    adapt_backend_module(np.add).require_symbol("at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: BackendArg, indices: BackendArg, updates: BackendArg) -> BackendArg:
    """Tensor scatter max for numpy.

    Args:
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(tensor)
    )
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[BackendArg, ...] = tuple(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0))
    adapt_backend_module(np.maximum).require_symbol("at")(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: BackendArg, indices: BackendArg, updates: BackendArg) -> BackendArg:
    """Tensor scatter min for numpy.

    Args:
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(tensor)
    )
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple: tuple[BackendArg, ...] = tuple(adapt_backend_module(np).require_op("moveaxis")(indices, -1, 0))
    adapt_backend_module(np.minimum).require_symbol("at")(res, idx_tuple, updates)
    return res


def _np_take_along_axis(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    indices: BackendArg,
    axis: BackendArg,
) -> BackendArg:
    """Execute the eager np take along axis operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        indices (BackendArg): Input `indices` argument (indices).
        axis (BackendArg): Input `axis` argument (axis).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("take_along_axis")(x, indices, axis=axis)


def _np_tensor_scatter_update(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> BackendArg:
    """Execute the eager np tensor scatter update operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return require_callable(global_eager_registry.get)("TensorScatterUpdate")(backend_module, tensor, indices, updates)


def _np_tensor_scatter_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> BackendArg:
    """Execute the eager np tensor scatter add operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[BackendArg, ...] = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("add").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return res


def _np_tensor_scatter_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> BackendArg:
    """Execute the eager np tensor scatter max operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[BackendArg, ...] = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("maximum").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return res


def _np_tensor_scatter_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> BackendArg:
    """Execute the eager np tensor scatter min operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[BackendArg, ...] = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("minimum").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return res


def _np_gather_nd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, params: BackendArg, indices: BackendArg
) -> BackendArg:
    """Execute the eager np gather nd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        params (BackendArg): Input `params` argument (params).
        indices (BackendArg): Input `indices` argument (indices).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    return coerce_array(backend_module, params)[idx]


def _np_scatter_nd(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    indices: BackendArg,
    updates: BackendArg,
    shape: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np scatter nd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).
        shape (BackendArg): Input `shape` argument (shape).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("zeros")(shape, dtype=adapt_backend_module(updates).require_symbol("dtype"))
    )
    moved = ensure_ndarray(adapt_backend_module(np).require_op("moveaxis")(np.array(indices), -1, 0)).astype(np.intp)
    idx: tuple[np.ndarray[tuple[int, ...], np.dtype[np.intp]], ...] = tuple(
        np.asarray(moved[i], dtype=np.intp) for i in range(moved.shape[0])
    )
    out[idx] = ensure_ndarray(updates)
    return out


def _np_scatter(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scatter operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[0])
    index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[2])
    dim: int = coerce_int(kwargs.get("dim", 0))
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.copy(input_data)
    adapt_backend_module(np).require_op("put_along_axis")(out, index, src, axis=dim)
    return out


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scatter add operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    input_data: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(args[0])
    )
    index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[1])
    src: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[2])
    dim: int = coerce_int(kwargs.get("dim", 0))
    adapt_backend_module(np).require_op("put_along_axis")(
        input_data,
        index,
        np.add(
            ensure_ndarray(adapt_backend_module(np).require_op("take_along_axis")(input_data, index, axis=dim)), src
        ),
        axis=dim,
    )
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    context: IndexingContext | BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np scatter apply operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        context (IndexingContext | BackendArg): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

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
    reduction: str | None = coerce_opt_str(kwargs.get("reduction", None))

    try:
        if reduction == "add":
            np.add.at(tensor, tuple(indices.T), updates)
        elif reduction == "mul":
            np.multiply.at(tensor, tuple(indices.T), updates)
        else:
            tensor[tuple(indices.T)] = updates
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise RuntimeError(f"Eager execution failed in ScatterNd: {e}") from e
    return tensor


@numpy_eager_registry.register("ScatterMax")
def _np_scatter_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
    context: threading.Thread | IndexingContext | None = None,
) -> BackendArg:
    """Execute the eager np scatter max operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).
        context (threading.Thread | IndexingContext | None): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr = np.copy(np.asarray(tensor))
    np.maximum.at(arr, tuple(np.asarray(indices).T), np.asarray(updates))
    return arr


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
    context: threading.Thread | IndexingContext | None = None,
) -> BackendArg:
    """Execute the eager np scatter min operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).
        context (threading.Thread | IndexingContext | None): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_min = np.copy(np.asarray(tensor))
    np.minimum.at(arr_min, tuple(np.asarray(indices).T), np.asarray(updates))
    return arr_min


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
    context: threading.Thread | IndexingContext | None = None,
) -> BackendArg:
    """Execute the eager np scatter mul operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        tensor (BackendArg): Input `tensor` argument (tensor).
        indices (BackendArg): Input `indices` argument (indices).
        updates (BackendArg): Input `updates` argument (updates).
        context (threading.Thread | IndexingContext | None): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mul = np.copy(np.asarray(tensor))
    np.multiply.at(arr_mul, tuple(np.asarray(indices).T), np.asarray(updates))
    return arr_mul
