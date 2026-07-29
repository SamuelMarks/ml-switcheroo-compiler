"""Advanced indexing and scatter/gather operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

from .indexing import IndexingContext


def _gather_nd(x: object, indices: object, **kwargs: object) -> object:
    """Evaluate."""
    return x[tuple(np.moveaxis(indices, -1, 0))]


def _scatter_nd(indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate."""
    res = np.zeros(shape, dtype=updates.dtype)
    res[tuple(np.moveaxis(indices, -1, 0))] = updates
    return res


def _scatter(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate."""
    y = np.copy(x)
    np.put_along_axis(y, index, src, axis=dim)
    return y


def _scatter_add(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate."""
    y = np.copy(x)
    it = np.nditer(index, flags=["multi_index"])
    for idx_val in it:
        pos = list(it.multi_index)
        pos[dim] = int(idx_val)
        y[tuple(pos)] += src[it.multi_index]
    return y


def _tensor_scatter_update(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter update for numpy."""
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    res[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter add for numpy."""
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.add.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter max for numpy."""
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.maximum.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter min for numpy."""
    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, -1, 0))
    np.minimum.at(res, idx_tuple, updates)
    return res


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


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the scatter add logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    input_data = np.copy(args[0])
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    np.put_along_axis(input_data, index, np.take_along_axis(input_data, index, axis=dim) + src, axis=dim)
    return input_data


@numpy_eager_registry.register("ScatterApply")
def _np_scatter_apply(backend_module: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate the scatter apply logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
    """Evaluate the scatter max logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.
        context (IndexingContext): Required parameter for context.

    Returns:
        object: The evaluated or processed output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.maximum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMin")
def _np_scatter_min(backend_module: object, tensor: object, indices: object, updates: object, context: IndexingContext = None) -> object:
    """Evaluate the scatter min logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.
        context (IndexingContext): Required parameter for context.

    Returns:
        object: The evaluated or processed output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.minimum.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor


@numpy_eager_registry.register("ScatterMul")
def _np_scatter_mul(backend_module: object, tensor: object, indices: object, updates: object, context: IndexingContext = None) -> object:
    """Evaluate the scatter mul logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        tensor (object): Required parameter for tensor.
        indices (object): Required parameter for indices.
        updates (object): Required parameter for updates.
        context (IndexingContext): Required parameter for context.

    Returns:
        object: The evaluated or processed output.
    """
    tensor = np.copy(np.asarray(tensor))
    np.multiply.at(tensor, tuple(np.asarray(indices).T), np.asarray(updates))
    return tensor
