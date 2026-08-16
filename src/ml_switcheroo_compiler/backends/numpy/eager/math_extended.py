# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy extra math operations."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window
from ml_switcheroo_compiler.core.dtype import DType


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(backend_module: Any, shape: Any, value: Any = 0.0, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_constant_of_shape operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        value (object): The value parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.full(shape, value)


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_reduce_window operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("TestEagerOp")
def _np_test_eager_op(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_test_eager_op operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.array([1, 2, 3], dtype=backend_module.float32)


@numpy_eager_registry.register("Unknown")
def _np_unknown(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unknown operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return 0.0


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rand operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_is_non_decreasing(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """IsNonDecreasing.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if backend_module.size(x) <= 1:
        return backend_module.array(True)
    diffs = backend_module.diff(x)
    return backend_module.all(diffs >= 0)


@numpy_eager_registry.register("IsStrictlyIncreasing")
def _np_is_strictly_increasing(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """IsStrictlyIncreasing.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if backend_module.size(x) <= 1:
        return backend_module.array(True)
    diffs = backend_module.diff(x)
    return backend_module.all(diffs > 0)


@numpy_eager_registry.register("L2Normalize")
def _np_l2_normalize(backend_module: Any, x: Any, axis: Any = None, epsilon: Any = 1e-12, **kwargs: Any) -> Any:
    """L2Normalize.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (int): The axis parameter.
        epsilon (float): The epsilon parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    square_sum = backend_module.sum(backend_module.square(x), axis=axis, keepdims=True)
    x_inv_norm = backend_module.divide(1.0, backend_module.sqrt(backend_module.maximum(square_sum, epsilon)))
    return backend_module.multiply(x, x_inv_norm)


@numpy_eager_registry.register("ReduceEuclideanNorm")
def _np_reduce_euclidean_norm(backend_module: Any, x: Any, axis: Any = None, keepdims: bool = False, **kwargs: Any) -> Any:
    """ReduceEuclideanNorm.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.sqrt(backend_module.sum(backend_module.square(x), axis=axis, keepdims=keepdims))


@numpy_eager_registry.register("Clamp")
def _clamp(np: Any, min_val: Any, x: Any, max_val: Any, **kwargs: Any) -> Any:
    """Clamp wrapper.

    Args:
        np (object): The np parameter.
        min_val (object): The min_val parameter.
        x (object): The x parameter.
        max_val (object): The max_val parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.clip(x, min_val, max_val)


@numpy_eager_registry.register("Logspace")
def _logspace(np: Any, start: Any, stop: Any, *args: Any, **kwargs: Any) -> Any:
    """Logspace wrapper.

    Args:
        np (object): The np parameter.
        start (object): The start parameter.
        stop (object): The stop parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if args and type(args[0]).__name__ == "SpaceConfig":
        c = args[0]
        return np.logspace(start, stop, num=c.num, endpoint=c.endpoint, base=c.base, dtype=c.dtype, axis=c.axis)
    return np.logspace(start, stop, *args, **kwargs)


@numpy_eager_registry.register("FromBuffer")
def _np_frombuffer(backend_module: Any, buffer: Any, dtype: str = "float32", count: int = -1, offset: int = 0, **kwargs: Any) -> Any:
    """Evaluate _np_frombuffer operation.

    Args:
        backend_module (object): The backend_module parameter.
        buffer (object): The buffer parameter.
        dtype (str): The dtype parameter.
        count (int): The count parameter.
        offset (int): The offset parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.frombuffer(buffer, dtype=dtype, count=count, offset=offset)


@numpy_eager_registry.register("DType")
def _np_dtype_op(backend_module: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    """DType.

    Args:
        backend_module (object): The backend_module parameter.
        value (object): The value parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if isinstance(value, str):
        return DType(value)
    if hasattr(value, "dtype"):
        return DType(str(value.dtype))
    return DType(str(backend_module.asarray(value).dtype))


@numpy_eager_registry.register("Gradient")
def _eager_Gradient(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eager_Gradient operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return getattr(backend_module, "gradient", lambda *a, **k: a[0])(*args, **kwargs)


@numpy_eager_registry.register("I0")
def _eager_I0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eager_I0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return getattr(backend_module, "i0", lambda *a, **k: a[0])(*args, **kwargs)


@numpy_eager_registry.register("BroadcastedIota")
def _eager_BroadcastedIota(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eager_BroadcastedIota operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return getattr(backend_module, "broadcast_to", lambda *a, **k: a[0])(backend_module.arange(args[0]), args[1])
