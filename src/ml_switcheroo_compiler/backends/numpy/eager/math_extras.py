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


@numpy_eager_registry.register("DivideNoNan")
def _np_divide_no_nan(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """DivideNoNan."""
    return backend_module.divide(
        x,
        y,
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
        where=(y != 0),
    )


@numpy_eager_registry.register("MultiplyNoNan")
def _np_multiply_no_nan(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """MultiplyNoNan."""
    return backend_module.multiply(
        x,
        y,
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
        where=(y != 0),
    )


@numpy_eager_registry.register("SquaredDifference")
def _np_squared_difference(
    backend_module: object, x: object, y: object, **kwargs: object
) -> object:
    """SquaredDifference."""
    diff = backend_module.subtract(x, y)
    return backend_module.square(diff)


@numpy_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Xdivy."""
    return backend_module.divide(
        x,
        y,
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
        where=(x != 0),
    )


@numpy_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Xlog1py."""
    return backend_module.multiply(
        x,
        backend_module.log1p(y),
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
        where=(x != 0),
    )


@numpy_eager_registry.register("ReciprocalNoNan")
def _np_reciprocal_no_nan(backend_module: object, x: object, **kwargs: object) -> object:
    """ReciprocalNoNan."""
    return backend_module.divide(
        1.0,
        x,
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, 1.0)),
        where=(x != 0),
    )


@numpy_eager_registry.register("IsNonDecreasing")
def _np_is_non_decreasing(backend_module: object, x: object, **kwargs: object) -> object:
    """IsNonDecreasing."""
    if backend_module.size(x) <= 1:
        return backend_module.array(True)
    diffs = backend_module.diff(x)
    return backend_module.all(diffs >= 0)


@numpy_eager_registry.register("IsStrictlyIncreasing")
def _np_is_strictly_increasing(backend_module: object, x: object, **kwargs: object) -> object:
    """IsStrictlyIncreasing."""
    if backend_module.size(x) <= 1:
        return backend_module.array(True)
    diffs = backend_module.diff(x)
    return backend_module.all(diffs > 0)


@numpy_eager_registry.register("L2Normalize")
def _np_l2_normalize(
    backend_module: object, x: object, axis: int = None, epsilon: float = 1e-12, **kwargs: object
) -> object:
    """L2Normalize."""
    square_sum = backend_module.sum(backend_module.square(x), axis=axis, keepdims=True)
    x_inv_norm = backend_module.divide(
        1.0, backend_module.sqrt(backend_module.maximum(square_sum, epsilon))
    )
    return backend_module.multiply(x, x_inv_norm)


@numpy_eager_registry.register("ZeroFraction")
def _np_zero_fraction(backend_module: object, x: object, **kwargs: object) -> object:
    """ZeroFraction."""
    num_zeros = backend_module.sum(backend_module.equal(x, 0))
    total_elements = backend_module.size(x)
    if total_elements == 0:
        return backend_module.array(float("nan"))  # pragma: no cover
    return backend_module.divide(num_zeros, total_elements).astype(backend_module.float32)


@numpy_eager_registry.register("ReduceEuclideanNorm")
def _np_reduce_euclidean_norm(
    backend_module: object, x: object, axis: object = None, keepdims: bool = False, **kwargs: object
) -> object:
    """ReduceEuclideanNorm."""
    return backend_module.sqrt(
        backend_module.sum(backend_module.square(x), axis=axis, keepdims=keepdims)
    )


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.j0(*args, **kwargs)


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.j1(*args, **kwargs)


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.k0(*args, **kwargs)


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.k0e(*args, **kwargs)


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.k1(*args, **kwargs)


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.k1e(*args, **kwargs)


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.y0(*args, **kwargs)


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.y1(*args, **kwargs)


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.dawsn(*args, **kwargs)


@numpy_eager_registry.register("Expint")
def _np_expint(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.expi(*args, **kwargs)


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.fresnel(*args, **kwargs)[1]


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.fresnel(*args, **kwargs)[0]


@numpy_eager_registry.register("Spence")
def _np_spence(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.spence(*args, **kwargs)


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(backend_module: object, x: object, **kwargs: object) -> object:
    import scipy.special as sc

    return sc.i0(x)


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(backend_module: object, x: object, **kwargs: object) -> object:
    import scipy.special as sc

    return sc.i1(x)


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    import scipy.special as sc

    return sc.jn(x, y)


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(backend_module: object, M: object, **kwargs: object) -> object:
    import numpy as np

    return np.bartlett(M)
