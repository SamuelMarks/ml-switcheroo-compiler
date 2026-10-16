"""Numpy extra math operations."""

from types import ModuleType
from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window
from ml_switcheroo_compiler.core.dtype import DType


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(backend_module: ModuleType, shape: EagerValue, value: EagerValue = 0.0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_constant_of_shape operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        value (object): The value parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.full(shape, value))


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_reduce_window operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _reduce_window)(*args, **kwargs))


@numpy_eager_registry.register("TestEagerOp")
def _np_test_eager_op(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_test_eager_op operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.array([1, 2, 3], dtype=backend_module.float32))


@numpy_eager_registry.register("Unknown")
def _np_unknown(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_unknown operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0.0


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_rand operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[-1]
    dt = getattr(backend_module, dtype_str, dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return cast(EagerValue, backend_module.array(backend_module.random.rand(*args)).astype(dt))


@numpy_eager_registry.register("IsNonDecreasing")
def _np_is_non_decreasing(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """IsNonDecreasing.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if backend_module.size(x) <= 1:
        return cast(EagerValue, backend_module.array(True))
    diffs = backend_module.diff(x)
    return cast(EagerValue, backend_module.all(diffs >= 0))


@numpy_eager_registry.register("IsStrictlyIncreasing")
def _np_is_strictly_increasing(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """IsStrictlyIncreasing.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if backend_module.size(x) <= 1:
        return cast(EagerValue, backend_module.array(True))
    diffs = backend_module.diff(x)
    return cast(EagerValue, backend_module.all(diffs > 0))


@numpy_eager_registry.register("L2Normalize")
def _np_l2_normalize(backend_module: ModuleType, x: EagerValue, axis: EagerValue = None, epsilon: EagerValue = 1e-12, **kwargs: EagerValue) -> EagerValue:
    """L2Normalize.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (int): The axis parameter.
        epsilon (float): The epsilon parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    square_sum = backend_module.sum(backend_module.square(x), axis=axis, keepdims=True)
    x_inv_norm = backend_module.divide(1.0, backend_module.sqrt(backend_module.maximum(square_sum, epsilon)))
    return cast(EagerValue, backend_module.multiply(x, x_inv_norm))


@numpy_eager_registry.register("ReduceEuclideanNorm")
def _np_reduce_euclidean_norm(backend_module: ModuleType, x: EagerValue, axis: EagerValue = None, keepdims: bool = False, **kwargs: EagerValue) -> EagerValue:
    """ReduceEuclideanNorm.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.sqrt(backend_module.sum(backend_module.square(x), axis=axis, keepdims=keepdims)))


@numpy_eager_registry.register("Clamp")
def _clamp(np: ModuleType, min_val: EagerValue, x: EagerValue, max_val: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Clamp wrapper.

    Args:
        np (object): The np parameter.
        min_val (object): The min_val parameter.
        x (object): The x parameter.
        max_val (object): The max_val parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.clip(x, min_val, max_val))


@numpy_eager_registry.register("Logspace")
def _logspace(np: ModuleType, start: EagerValue, stop: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Logspace wrapper.

    Args:
        np (object): The np parameter.
        start (object): The start parameter.
        stop (object): The stop parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if args and type(args[0]).__name__ == "SpaceConfig":
        c = args[0]
        return cast(EagerValue, np.logspace(start, stop, num=getattr(c, "num"), endpoint=getattr(c, "endpoint"), base=getattr(c, "base"), dtype=getattr(c, "dtype"), axis=getattr(c, "axis")))
    return cast(EagerValue, np.logspace(start, stop, *args, **kwargs))


@numpy_eager_registry.register("FromBuffer")
def _np_frombuffer(backend_module: ModuleType, buffer: EagerValue, dtype: str = "float32", count: int = -1, offset: int = 0, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_frombuffer operation.

    Args:
        backend_module (object): The backend_module parameter.
        buffer (object): The buffer parameter.
        dtype (str): The dtype parameter.
        count (int): The count parameter.
        offset (int): The offset parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.frombuffer(buffer, dtype=dtype, count=count, offset=offset))


@numpy_eager_registry.register("DType")
def _np_dtype_op(backend_module: ModuleType, value: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """DType.

    Args:
        backend_module (object): The backend_module parameter.
        value (object): The value parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if isinstance(value, str):
        return DType(value)
    if hasattr(value, "dtype"):
        return DType(str(getattr(value, "dtype")))
    return DType(str(backend_module.asarray(value).dtype))


@numpy_eager_registry.register("Gradient")
def _eager_Gradient(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _eager_Gradient operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(backend_module, "gradient", lambda *a, **k: a[0])(*args, **kwargs))


@numpy_eager_registry.register("I0")
def _eager_I0(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _eager_I0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(backend_module, "i0", lambda *a, **k: a[0])(*args, **kwargs))


@numpy_eager_registry.register("BroadcastedIota")
def _eager_BroadcastedIota(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _eager_BroadcastedIota operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(backend_module, "broadcast_to", lambda *a, **k: a[0])(backend_module.arange(args[0]), args[1]))
