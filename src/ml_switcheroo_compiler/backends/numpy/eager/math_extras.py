"""Numpy extra math operations."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import _top_k  # pragma: no cover
from ml_switcheroo_compiler.backends.numpy.eager.reductions import (
    _reduce_window,
)  # pragma: no cover
from ml_switcheroo_compiler.core.errors import CompilationError

from .shape import _band_part


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
def _np_resize(backend_module: object, x: object, shape: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        shape: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.zeros((x.shape[0], *shape, x.shape[-1]), dtype=x.dtype)  # pragma: no cover


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(backend_module: object, shape: object, value: object = 0.0, *args: object, **kwargs: object) -> object:
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
    return _top_k(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
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
def _np_assign(backend_module: object, x: object, y: object, *args: object, **kwargs: object) -> object:
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
def _np_cast(backend_module: object, x: object, dtype: object, *args: object, **kwargs: object) -> object:
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
def _np_bitcast(backend_module: object, x: object, dtype: object, *args: object, **kwargs: object) -> object:
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
def _np_unstack(backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        axis: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return [  # pragma: no cover
        backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, x.shape[axis], axis=axis)
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
    shape = args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape"))  # pragma: no cover
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


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    raise CompilationError("ReadVariable not supported in eager execution")


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    raise CompilationError("AssignVariable not supported in eager execution")


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
def _np_l2_normalize(backend_module: object, x: object, axis: int = None, epsilon: float = 1e-12, **kwargs: object) -> object:
    """L2Normalize."""
    square_sum = backend_module.sum(backend_module.square(x), axis=axis, keepdims=True)
    x_inv_norm = backend_module.divide(1.0, backend_module.sqrt(backend_module.maximum(square_sum, epsilon)))
    return backend_module.multiply(x, x_inv_norm)


@numpy_eager_registry.register("ReduceEuclideanNorm")
def _np_reduce_euclidean_norm(backend_module: object, x: object, axis: object = None, keepdims: bool = False, **kwargs: object) -> object:
    """ReduceEuclideanNorm."""
    return backend_module.sqrt(backend_module.sum(backend_module.square(x), axis=axis, keepdims=keepdims))
