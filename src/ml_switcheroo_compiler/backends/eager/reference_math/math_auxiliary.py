"""Generic backend-parametric eager reference operations for `reference_math.math_auxiliary`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.reference_math.reductions import _reduce_window
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.dtype import DType


@global_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    shape: BackendArg,
    value: BackendArg = 0.0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np constant of shape operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        shape (BackendArg): Input `shape` argument (shape).
        value (BackendArg): Input `value` argument (value).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("full")(shape, value)


@global_eager_registry.register("ReduceWindow")
def _np_reduce_window(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np reduce window operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    return require_callable(_reduce_window)(*args, **kwargs)


@global_eager_registry.register("TestEagerOp")
def _np_test_eager_op(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np test eager op operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(
        [1, 2, 3], dtype=adapt_backend_module(backend_module).require_symbol("float32")
    )


@global_eager_registry.register("Unknown")
def _np_unknown(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np unknown operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return 0.0


@global_eager_registry.register("Rand")
def _np_rand(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rand operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dtype = kwargs.get("dtype", adapt_backend_module(backend_module).get_symbol("float32", None))
    dtype_str = str(dtype).split(".")[-1]
    dt = adapt_backend_module(backend_module).get_symbol(dtype_str, dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return (
        adapt_backend_module(backend_module)
        .require_symbol("array")(adapt_backend_module(backend_module).require_symbol("random").rand(*args))
        .astype(dt)
    )


@global_eager_registry.register("IsNonDecreasing")
def _np_is_non_decreasing(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """IsNonDecreasing.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if adapt_backend_module(backend_module).require_symbol("size")(x) <= 1:
        return adapt_backend_module(backend_module).require_symbol("array")(True)
    diffs = adapt_backend_module(backend_module).require_symbol("diff")(x)
    return adapt_backend_module(backend_module).require_symbol("all")(diffs >= 0)


@global_eager_registry.register("IsStrictlyIncreasing")
def _np_is_strictly_increasing(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """IsStrictlyIncreasing.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if adapt_backend_module(backend_module).require_symbol("size")(x) <= 1:
        return adapt_backend_module(backend_module).require_symbol("array")(True)
    diffs = adapt_backend_module(backend_module).require_symbol("diff")(x)
    return adapt_backend_module(backend_module).require_symbol("all")(diffs > 0)


@global_eager_registry.register("L2Normalize")
def _np_l2_normalize(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: BackendArg = None,
    epsilon: BackendArg = 1e-12,
    **kwargs: BackendArg,
) -> BackendArg:
    """L2Normalize.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axis (BackendArg): Input `axis` argument (axis).
        epsilon (BackendArg): Input `epsilon` argument (epsilon).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    square_sum = adapt_backend_module(backend_module).require_symbol("sum")(
        adapt_backend_module(backend_module).require_symbol("square")(x), axis=axis, keepdims=True
    )
    x_inv_norm = adapt_backend_module(backend_module).require_symbol("divide")(
        1.0,
        adapt_backend_module(backend_module).require_symbol("sqrt")(
            adapt_backend_module(backend_module).require_symbol("maximum")(square_sum, epsilon)
        ),
    )
    return adapt_backend_module(backend_module).require_symbol("multiply")(x, x_inv_norm)


@global_eager_registry.register("ReduceEuclideanNorm")
def _np_reduce_euclidean_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: BackendArg = None,
    keepdims: bool = False,
    **kwargs: BackendArg,
) -> BackendArg:
    """ReduceEuclideanNorm.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axis (BackendArg): Input `axis` argument (axis).
        keepdims (bool): Input `keepdims` argument (keepdims).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("sqrt")(
        adapt_backend_module(backend_module).require_symbol("sum")(
            adapt_backend_module(backend_module).require_symbol("square")(x), axis=axis, keepdims=keepdims
        )
    )


@global_eager_registry.register("Clamp")
def _clamp(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    min_val: BackendArg,
    x: BackendArg,
    max_val: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Clamp wrapper.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        min_val (BackendArg): Input `min_val` argument (min val).
        x (BackendArg): Input `x` argument (x).
        max_val (BackendArg): Input `max_val` argument (max val).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(arr_mod).require_symbol("clip")(x, min_val, max_val)


@global_eager_registry.register("Logspace")
def _logspace(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    start: BackendArg,
    stop: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Logspace wrapper.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        start (BackendArg): Input `start` argument (start).
        stop (BackendArg): Input `stop` argument (stop).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if args and type(args[0]).__name__ == "SpaceConfig":
        c = args[0]
        return adapt_backend_module(arr_mod).require_symbol("logspace")(
            start,
            stop,
            num=adapt_backend_module(c).require_symbol("num"),
            endpoint=adapt_backend_module(c).require_symbol("endpoint"),
            base=adapt_backend_module(c).require_symbol("base"),
            dtype=adapt_backend_module(c).require_symbol("dtype"),
            axis=adapt_backend_module(c).require_symbol("axis"),
        )
    return adapt_backend_module(arr_mod).require_symbol("logspace")(start, stop, *args, **kwargs)


@global_eager_registry.register("FromBuffer")
def _np_frombuffer(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    buffer: BackendArg,
    dtype: str = "float32",
    count: int = -1,
    offset: int = 0,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np frombuffer operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        buffer (BackendArg): Input `buffer` argument (buffer).
        dtype (str): Input `dtype` argument (dtype).
        count (int): Input `count` argument (count).
        offset (int): Input `offset` argument (offset).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("frombuffer")(
        buffer, dtype=dtype, count=count, offset=offset
    )


@global_eager_registry.register("DType")
def _np_dtype_op(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    value: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """DType.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        value (BackendArg): Input `value` argument (value).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(value, str):
        return DType(value)
    if adapt_backend_module(value).has_symbol("dtype"):
        return DType(str(adapt_backend_module(value).require_symbol("dtype")))
    return DType(str(adapt_backend_module(backend_module).require_symbol("asarray")(value).dtype))


@global_eager_registry.register("Gradient")
def _eager_Gradient(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eager Gradient operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    grad_fn = adapt_backend_module(backend_module).get_op("gradient")
    return grad_fn(*args, **kwargs) if grad_fn is not None else args[0]


@global_eager_registry.register("I0")
def _eager_I0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eager I0 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    i0_fn = adapt_backend_module(backend_module).get_op("i0")
    return i0_fn(*args, **kwargs) if i0_fn is not None else args[0]


@global_eager_registry.register("BroadcastedIota")
def _eager_BroadcastedIota(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eager BroadcastedIota operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    iota_arr = adapt_backend_module(backend_module).require_symbol("arange")(args[0])
    bcast_fn = adapt_backend_module(backend_module).get_op("broadcast_to")
    return bcast_fn(iota_arr, args[1]) if bcast_fn is not None else iota_arr
