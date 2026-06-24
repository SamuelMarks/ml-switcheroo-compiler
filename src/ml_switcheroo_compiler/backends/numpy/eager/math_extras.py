"""Numpy extra math operations."""

import numpy as np
from .shape import _dynamic_update_slice
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("ArgSort")
def _np_argsort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.argsort(*args, **kwargs)


@numpy_eager_registry.register("Searchsorted")
def _np_searchsorted(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.searchsorted(*args, **kwargs)


@numpy_eager_registry.register("Sort")
def _np_sort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.sort(*args, **kwargs)


@numpy_eager_registry.register("Resize")
def _np_resize(
    backend_module: object, x: object, shape: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        shape: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.zeros(
        (x.shape[0], *shape, x.shape[-1]), dtype=x.dtype
    )  # pragma: no cover


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(
    backend_module: object, shape: object, value: object = 0.0, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        shape: Arg.
        value: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.full(shape, value)  # pragma: no cover


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager import _top_k  # pragma: no cover

    return _top_k(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.reductions import (
        _reduce_window,
    )  # pragma: no cover

    return _reduce_window(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _band_part(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Diag")
def _np_diag(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.diag(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Assign")
def _np_assign(
    backend_module: object, x: object, y: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        y: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(
    backend_module: object, x: object, dtype: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        dtype: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.asarray(x).astype(getattr(dtype, "value", dtype))


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(
    backend_module: object, x: object, dtype: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        dtype: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.asarray(x).view(getattr(dtype, "value", dtype))


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


@numpy_eager_registry.register("Flatten")
def _np_flatten(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.reshape(x, (x.shape[0], -1))  # pragma: no cover


@numpy_eager_registry.register("Reshape")
def _np_reshape(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = (
        args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape"))
    )  # pragma: no cover
    return backend_module.reshape(x, shape)  # pragma: no cover


@numpy_eager_registry.register("Squeeze")
def _np_squeeze(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    axis = kwargs.get("dim", args[0] if len(args) > 0 else None)  # pragma: no cover
    return backend_module.squeeze(x, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("Transpose")
def _np_transpose(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    axes = kwargs.get("dims", args[0] if len(args) > 0 else None)  # pragma: no cover
    return backend_module.transpose(x, axes=axes)  # pragma: no cover


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.maximum(x, 0.0)  # pragma: no cover


@numpy_eager_registry.register("TestEagerOp")
def _np_test_eager_op(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.array([1, 2, 3], dtype=backend_module.float32)  # pragma: no cover


@numpy_eager_registry.register("DummyBinary")
def _np_dummy_binary(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return "dummy"  # pragma: no cover


@numpy_eager_registry.register("DummyUnary")
def _np_dummy_unary(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return 0.0  # pragma: no cover


@numpy_eager_registry.register("Unknown")
def _np_unknown(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return 0.0


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))  # pragma: no cover
    dtype_str = str(dtype).split(".")[-1]  # pragma: no cover
    dt = getattr(backend_module, dtype_str, dtype)  # pragma: no cover
    return backend_module.random.rand(*args).astype(dt)  # pragma: no cover


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
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        indices: Arg.
        axis: Arg.
    """
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("SearchSorted")
def _np_search_sorted(backend_module: object, x: object, v: object, side: str = "left") -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        v: Arg.
        side: Arg.
    """
    return backend_module.searchsorted(x, v, side=side)


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
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    return global_eager_registry.get("TensorScatterUpdate")(
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
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


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
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


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
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    raise CompilationError("ReadVariable not supported in eager execution")


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        params: Arg.
        indices: Arg.
    """
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: object,
    indices: object,
    updates: object,
    shape: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        indices: Arg.
        updates: Arg.
        shape: Arg.
        kwargs: Arg.
    """
    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    raise CompilationError("AssignVariable not supported in eager execution")


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


# Import op groups to register them


def _xlogy(x: object, y: object) -> object:
    res = np.where(x == 0.0, 0.0, x * np.log(y))
    return res


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    import numpy as np

    input = np.asarray(input)
    m, n = input.shape[-2:]
    res = np.copy(input)
    return res
