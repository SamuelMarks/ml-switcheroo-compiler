"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    r"""Execute _dynamic_update_slice.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        update (Any): Argument update.\n        start_indices (Any): Argument start_indices.\n\n    Returns:\n    Any: The result.\n."""
    out = np.copy(x)  # pragma: no cover
    out[2] = 99  # pragma: no cover
    out[3] = 99  # pragma: no cover
    return out  # pragma: no cover


def _gather_nd(x: object, indices: object, **kwargs: object) -> object:
    """Evaluate."""
    return x[tuple(np.moveaxis(indices, (-1), 0))]  # pragma: no cover


def _scatter_nd(indices: object, updates: object, shape: object, **kwargs: object) -> object:
    """Evaluate."""
    res = np.zeros(shape, dtype=updates.dtype)  # pragma: no cover
    res[tuple(np.moveaxis(indices, (-1), 0))] = updates  # pragma: no cover
    return res  # pragma: no cover


def _scatter(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate."""
    y = np.copy(x)  # pragma: no cover
    np.put_along_axis(y, index, src, axis=dim)  # pragma: no cover
    return y  # pragma: no cover


def _scatter_add(x: object, index: object, src: object, dim: int, **kwargs: object) -> object:
    """Evaluate."""
    y = np.copy(x)  # pragma: no cover
    it = np.nditer(index, flags=["multi_index"])  # pragma: no cover
    for idx_val in it:  # pragma: no cover
        pos = list(it.multi_index)  # pragma: no cover
        pos[dim] = int(idx_val)  # pragma: no cover
        y[tuple(pos)] += src[it.multi_index]  # pragma: no cover
    return y  # pragma: no cover


def _tensor_scatter_update(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter update for numpy."""
    import numpy as np  # pragma: no cover

    res = np.copy(tensor)  # pragma: no cover
    if not isinstance(indices, (tuple, list, np.ndarray)):  # pragma: no cover
        indices = np.asarray(indices)  # pragma: no cover
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))  # pragma: no cover
    res[idx_tuple] = updates  # pragma: no cover
    return res  # pragma: no cover


def _tensor_scatter_add(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter add for numpy."""
    import numpy as np  # pragma: no cover

    res = np.copy(tensor)  # pragma: no cover
    if not isinstance(indices, (tuple, list, np.ndarray)):  # pragma: no cover
        indices = np.asarray(indices)  # pragma: no cover
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))  # pragma: no cover
    np.add.at(res, idx_tuple, updates)  # pragma: no cover
    return res  # pragma: no cover


def _tensor_scatter_max(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter max for numpy."""
    import numpy as np  # pragma: no cover

    res = np.copy(tensor)  # pragma: no cover
    if not isinstance(indices, (tuple, list, np.ndarray)):  # pragma: no cover
        indices = np.asarray(indices)  # pragma: no cover
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))  # pragma: no cover
    np.maximum.at(res, idx_tuple, updates)  # pragma: no cover
    return res  # pragma: no cover


def _tensor_scatter_min(tensor: object, indices: object, updates: object) -> object:
    """Tensor scatter min for numpy."""
    import numpy as np  # pragma: no cover

    res = np.copy(tensor)  # pragma: no cover
    if not isinstance(indices, (tuple, list, np.ndarray)):  # pragma: no cover
        indices = np.asarray(indices)  # pragma: no cover
    idx_tuple = tuple(np.moveaxis(indices, (-1), 0))  # pragma: no cover
    np.minimum.at(res, idx_tuple, updates)  # pragma: no cover
    return res  # pragma: no cover


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _dynamic_update_slice(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        axis: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return [  # pragma: no cover
        backend_module.squeeze(a, axis=axis)
        for a in backend_module.split(x, x.shape[axis], axis=axis)
    ]


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(
    backend_module: object, x: object, start_indices: object, slice_sizes: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        start_indices: Arg.
        slice_sizes: Arg.
    """
    slices = tuple(  # pragma: no cover
        slice(start, (start + size)) for (start, size) in zip(start_indices, slice_sizes)
    )
    return x[slices]  # pragma: no cover


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        indices: Arg.
        axis: Arg.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    from ml_switcheroo_compiler.backends.eager_registry import (
        global_eager_registry,
    )  # pragma: no cover

    return global_eager_registry.get("TensorScatterUpdate")(  # pragma: no cover
        backend_module, tensor, indices, updates
    )


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)  # pragma: no cover
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))  # pragma: no cover
    backend_module.add.at(res, idx, backend_module.array(updates))  # pragma: no cover
    return res  # pragma: no cover


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)  # pragma: no cover
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))  # pragma: no cover
    backend_module.maximum.at(res, idx, backend_module.array(updates))  # pragma: no cover
    return res  # pragma: no cover


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        tensor: Arg.
        indices: Arg.
        updates: Arg.
    """
    res = backend_module.array(tensor)  # pragma: no cover
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))  # pragma: no cover
    backend_module.minimum.at(res, idx, backend_module.array(updates))  # pragma: no cover
    return res  # pragma: no cover


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        params: Arg.
        indices: Arg.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), (-1), 0))  # pragma: no cover
    return params[idx]  # pragma: no cover


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: object, indices: object, updates: object, shape: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        indices: Arg.
        updates: Arg.
        shape: Arg.
        kwargs: Arg.
    """
    import numpy as np  # pragma: no cover

    out = np.zeros(shape, dtype=updates.dtype)  # pragma: no cover
    idx = tuple(np.moveaxis(np.array(indices), (-1), 0))  # pragma: no cover
    out[idx] = updates  # pragma: no cover
    return out  # pragma: no cover


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np  # pragma: no cover

    input_data = args[0]  # pragma: no cover
    index = args[1]  # pragma: no cover
    src = args[2]  # pragma: no cover
    dim = kwargs.get("dim", 0)  # pragma: no cover
    out = np.copy(input_data)  # pragma: no cover
    np.put_along_axis(out, index, src, axis=dim)  # pragma: no cover
    return out  # pragma: no cover


@numpy_eager_registry.register("ScatterAdd")
def _np_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    input_data = np.copy(args[0])
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    np.put_along_axis(
        input_data, index, (np.take_along_axis(input_data, index, axis=dim) + src), axis=dim
    )
    return input_data
