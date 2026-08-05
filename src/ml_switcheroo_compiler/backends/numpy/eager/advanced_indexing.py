"""Advanced indexing and scatter/gather operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .indexing import IndexingContext


def _gather_nd(x: object, indices: object, **kwargs: object) -> object:
    """Evaluate.

    Args:
        x (object): The x parameter.
        indices (object): The indices parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return x[tuple(np.moveaxis(indices, -1, 0))]


def _scatter_nd(indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate.

    Args:
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    res = np.zeros(shape, dtype=updates.dtype)
    res[tuple(np.moveaxis(indices, -1, 0))] = updates
    return res


def _scatter(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate.

    Args:
        x (object): The x parameter.
        index (object): The index parameter.
        src (object): The src parameter.
        dim (int): The dim parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    y = np.copy(x)
    np.put_along_axis(y, index, src, axis=dim)
    return y


def _scatter_add(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate.

    Args:
        x (object): The x parameter.
        index (object): The index parameter.
        src (object): The src parameter.
        dim (int): The dim parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    y = np.copy(x)
    it = np.nditer(index, flags=["multi_index"])
    for idx_val in it:
        pos = list(it.multi_index)
        pos[dim] = int(idx_val)
        y[tuple(pos)] += src[it.multi_index]
    return y


def _tensor_scatter_update(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter update for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    res[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter add for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.add.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter max for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.maximum.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter min for numpy.

    Args:
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.

    Returns:
        object: Result.
    """
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.minimum.at(res, idx_tuple, updates)
    return res


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


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_scatter_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    input_data = np.copy(args[0])
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    np.put_along_axis(input_data, index, np.take_along_axis(input_data, index, axis=dim) + src, axis=dim)
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(backend_module: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_scatter_apply operation.

    Args:
        backend_module (object): The backend_module parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

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
def _np_scatter_max(backend_module: object, tensor: object, indices: object, updates: object, context: IndexingContext = None) -> object:
    """Evaluate _np_scatter_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
        object: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.maximum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(backend_module: object, tensor: object, indices: object, updates: object, context: IndexingContext = None) -> object:
    """Evaluate _np_scatter_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
        object: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.minimum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(backend_module: object, tensor: object, indices: object, updates: object, context: IndexingContext = None) -> object:
    """Evaluate _np_scatter_mul operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        indices (object): The indices parameter.
        updates (object): The updates parameter.
        context (IndexingContext): The context parameter.

    Returns:
        object: Result.
    """
    tensor = np.copy(np.asarray(tensor))
    np.multiply.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor
