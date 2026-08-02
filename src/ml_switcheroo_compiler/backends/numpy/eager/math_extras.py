# ruff: noqa: E501
"""Numpy extra math operations."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window
from ml_switcheroo_compiler.core.dtype import DType


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(backend_module: object, shape: object, value: object = 0.0, *args: object, **kwargs: object) -> object:
    """Evaluate the constant of shape logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        shape (object): Required parameter for shape.
        value (object): Required parameter for value.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.full(shape, value)


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the reduce window logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("TestEagerOp")
def _np_test_eager_op(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the test eager op logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.array([1, 2, 3], dtype=backend_module.float32)


@numpy_eager_registry.register("Unknown")
def _np_unknown(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the unknown logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return 0.0


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rand logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[-1]
    dt = getattr(backend_module, dtype_str, dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return backend_module.array(backend_module.random.rand(*args)).astype(dt)


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


@numpy_eager_registry.register("Clamp")
def _clamp(np: object, min_val: object, x: object, max_val: object, **kwargs: object) -> object:
    """Clamp wrapper."""
    return np.clip(x, min_val, max_val)


@numpy_eager_registry.register("Logspace")
def _logspace(np: object, start: object, stop: object, *args: object, **kwargs: object) -> object:
    """Logspace wrapper."""
    if args and type(args[0]).__name__ == "SpaceConfig":
        c = args[0]
        return np.logspace(start, stop, num=c.num, endpoint=c.endpoint, base=c.base, dtype=c.dtype, axis=c.axis)
    return np.logspace(start, stop, *args, **kwargs)


@numpy_eager_registry.register("FromBuffer")
def _np_frombuffer(backend_module: object, buffer: object, dtype: str = "float32", count: int = -1, offset: int = 0, **kwargs: object) -> object:
    """Evaluate the frombuffer logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        buffer (object): Required parameter for buffer.
        dtype (str): Required parameter for dtype.
        count (int): Required parameter for count.
        offset (int): Required parameter for offset.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.frombuffer(buffer, dtype=dtype, count=count, offset=offset)


@numpy_eager_registry.register("DType")
def _np_dtype_op(backend_module: object, value: object, *args: object, **kwargs: object) -> object:
    """DType."""
    if isinstance(value, str):
        return DType(value)
    if hasattr(value, "dtype"):
        return DType(str(value.dtype))
    return DType(str(backend_module.asarray(value).dtype))


@numpy_eager_registry.register("Gradient")
def _eager_Gradient(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the Gradient logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return getattr(backend_module, "gradient", lambda *a, **k: a[0])(*args, **kwargs)


@numpy_eager_registry.register("I0")
def _eager_I0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the I0 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return getattr(backend_module, "i0", lambda *a, **k: a[0])(*args, **kwargs)


@numpy_eager_registry.register("BroadcastedIota")
def _eager_BroadcastedIota(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the BroadcastedIota logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return getattr(backend_module, "broadcast_to", lambda *a, **k: a[0])(backend_module.arange(args[0]), args[1])
