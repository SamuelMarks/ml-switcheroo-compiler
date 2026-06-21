"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    r"""Execute _dynamic_update_slice.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        update (Any): Argument update.\n        start_indices (Any): Argument start_indices.\n\n    Returns:\n    Any: The result.\n."""
    out = np.copy(x)
    out[2] = 99
    out[3] = 99
    return out


def _gather_nd(x: object, indices: object, **kwargs: object) -> object:
    """Evaluate."""
    return x[tuple(np.moveaxis(indices, (-1), 0))]


def _scatter_nd(indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate."""
    res = np.zeros(shape, dtype=updates.dtype)
    res[tuple(np.moveaxis(indices, (-1), 0))] = updates
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
    import numpy as np

    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))
    res[idx_tuple] = updates
    return res


def _tensor_scatter_add(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter add for numpy."""
    import numpy as np

    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))
    np.add.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_max(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter max for numpy."""
    import numpy as np

    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))
    np.maximum.at(res, idx_tuple, updates)
    return res


def _tensor_scatter_min(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter min for numpy."""
    import numpy as np

    res = np.copy(tensor)
    if not isinstance(indices, (tuple, list, np.ndarray)):
        indices = np.asarray(indices)
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))
    np.minimum.at(res, idx_tuple, updates)
    return res


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:

    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object
) -> object:
    return [
        backend_module.squeeze(a, axis=axis)
        for a in backend_module.split(x, x.shape[axis], axis=axis)
    ]


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(
    backend_module: object, x: object, start_indices: object, slice_sizes: object
) -> object:
    slices = tuple(
        slice(start, (start + size)) for (start, size) in zip(start_indices, slice_sizes)
    )
    return x[slices]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    return global_eager_registry.get("TensorScatterUpdate")(
        backend_module, tensor, indices, updates
    )


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))
    return params[idx]


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: object, indices: object, updates: object, shape: object, **kwargs: object
) -> object:
    import numpy as np

    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), (-1), 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    input_data = np.copy(args[0])
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    np.put_along_axis(
        input_data, index, (np.take_along_axis(input_data, index, axis=dim) + src), axis=dim
    )
    return input_data
