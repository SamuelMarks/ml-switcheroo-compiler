# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Advanced indexing and scatter/gather operations."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .indexing import IndexingContext


def _gather_nd(x: Any, indices: Any, **kwargs: Any) -> Any:
    """Evaluate.

    Args:
        x (object): The x parameter.
        indices (object): The indices parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return x[tuple(np.moveaxis(indices, -1, 0))]


def _scatter_nd(indices: Any, updates: Any, shape: Any, **kwargs: Any) -> Any:
    """Evaluate.

    Args:
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    res = np.zeros(shape, dtype=updates.dtype)
    res[tuple(np.moveaxis(indices, -1, 0))] = updates
    return res


def _scatter(x: Any, index: Any, src: Any, dim: int, **kwargs: Any) -> Any:
    """Evaluate.

    Args:
        x (object): The x parameter.
        index (object): The index parameter.
        src (object): The src parameter.
        dim (int): The dim parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    y = np.copy(x)
    np.put_along_axis(y, index, src, axis=dim)
    return y


def _scatter_add(x: Any, index: Any, src: Any, dim: int, **kwargs: Any) -> Any:
    """Evaluate.

    Args:
        x (object): The x parameter.
        index (object): The index parameter.
        src (object): The src parameter.
        dim (int): The dim parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    y = np.copy(x)
    it = np.nditer(index, flags=["multi_index"])
    for idx_val in it:
        pos = list(it.multi_index)
        pos[dim] = int(np.asarray(idx_val).item())
        y[tuple(pos)] += src[it.multi_index]
    return y


def _tensor_scatter_update(tensor: Any, indices: Any, updates: Any) -> Any:
    """Tensor scatter update for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    res[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: Any, indices: Any, updates: Any) -> Any:
    """Tensor scatter add for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.add.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: Any, indices: Any, updates: Any) -> Any:
    """Tensor scatter max for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.maximum.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: Any, indices: Any, updates: Any) -> Any:
    """Tensor scatter min for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.minimum.at(res, idx_tuple, updates)
    return res


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: Any, x: Any, indices: Any, axis: Any) -> Any:
    """Evaluate _np_take_along_axis operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        indices (object): The indices parameter.
        axis (object): The axis parameter.

    Returns: Any: Result.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(backend_module: Any, tensor: Any, indices: Any, updates: Any) -> Any:
    """Evaluate _np_tensor_scatter_update operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    return global_eager_registry.get("TensorScatterUpdate")(backend_module, tensor, indices, updates)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(backend_module: Any, tensor: Any, indices: Any, updates: Any) -> Any:
    """Evaluate _np_tensor_scatter_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(backend_module: Any, tensor: Any, indices: Any, updates: Any) -> Any:
    """Evaluate _np_tensor_scatter_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(backend_module: Any, tensor: Any, indices: Any, updates: Any) -> Any:
    """Evaluate _np_tensor_scatter_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns: Any: Result.
    """
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: Any, params: Any, indices: Any) -> Any:
    """Evaluate _np_gather_nd operation.

    Args:
        backend_module (object): The backend_module parameter.
        params (object): The params parameter.
        indices (object): The indices parameter.

    Returns: Any: Result.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(backend_module: Any, indices: Any, updates: Any, shape: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatter_nd operation.

    Args:
        backend_module (object): The backend_module parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatter_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    input_data = np.copy(args[0])
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    np.put_along_axis(input_data, index, np.take_along_axis(input_data, index, axis=dim) + src, axis=dim)
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(backend_module: Any, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatter_apply operation.

    Args:
        backend_module (object): The backend_module parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    if len(args) == 0:
        return None
    tensor = np.asarray(args[0]).copy()
    if len(args) < 3:
        return tensor

    indices = np.asarray(args[1])
    updates = np.asarray(args[2])
    reduction = kwargs.get("reduction", None)

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
def _np_scatter_max(backend_module: Any, tensor: Any, indices: Any, updates: Any, context: Any = None) -> Any:
    """Evaluate _np_scatter_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns: Any: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.maximum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(backend_module: Any, tensor: Any, indices: Any, updates: Any, context: Any = None) -> Any:
    """Evaluate _np_scatter_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns: Any: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.minimum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(backend_module: Any, tensor: Any, indices: Any, updates: Any, context: Any = None) -> Any:
    """Evaluate _np_scatter_mul operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns: Any: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.multiply.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor
