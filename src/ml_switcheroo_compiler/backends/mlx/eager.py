"""Backend utilities."""

from __future__ import annotations

import builtins
from types import ModuleType
from typing import Callable, cast

try:
    import mlx.core as mx
except ImportError:
    mx = cast(ModuleType, None)

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
    mlx_eager_registry,
)


def _get_mlx_func(op_type: str) -> EagerValue:
    """Retrieve the corresponding MLX function for a given operation type.

    Args:
        op_type (str): The name of the operation.

    Returns: mx.array: The MLX function if found, otherwise None.
    """
    import re

    s1: str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake: str = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    if snake == "mul":
        snake = "multiply"
    elif snake == "sub":
        snake = "subtract"
    elif snake == "div":
        snake = "divide"
    for mod in [mx, getattr(mx, "linalg", None), getattr(mx, "fft", None)]:
        if mod is not None and hasattr(mod, snake):
            return cast(EagerValue, getattr(mod, snake))
    return None


def execute_op(cls: type, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute op operation on the active backend array.

    Args:
        cls (type): The class.
        op_type (str): The op_type parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        BackendNotSupportedError: An exception.
    """
    try:
        func_registry = mlx_eager_registry.get(op_type)
        if func_registry is not None:
            return func_registry(mx, *args, **kwargs)
        func_registry = global_eager_registry.get(op_type)
        if func_registry is not None:
            return func_registry(mx, *args, **kwargs)
        func = _get_mlx_func(op_type)
        if func is None:
            raise AttributeError(f"No attribute found for {op_type}")
        return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", func)(*args, **kwargs))
    except AttributeError:
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        raise BackendNotSupportedError("Operation " + op_type + " is not implemented for the MLX backend.") from None


@mlx_eager_registry.register("Cast")
def _mlx_cast(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx cast operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor = args[0]
    dtype_val = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return tensor
    return cast(EagerValue, getattr(tensor, "astype")(dtype))


@mlx_eager_registry.register("Relu")
def _mlx_relu(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ReLU activation using MLX.

    Args:
        backend_module (ModuleType): The MLX core module.
        x (EagerValue): Input tensor array.
        *args (EagerValue): Additional positional arguments.
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Rectified linear activated tensor array.
    """
    try:
        import mlx.nn as nn

        return cast(EagerValue, getattr(nn, "relu")(x))
    except (ImportError, AttributeError):
        return cast(EagerValue, backend_module.maximum(x, 0))


@mlx_eager_registry.register("RaggedTensorToDense")
def _mlx_ragged_tensor_to_dense(backend_module: ModuleType, rt_input: EagerValue, default_value: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Convert a ragged tensor representation to a dense padded array using MLX.

    Args:
        backend_module (ModuleType): The MLX backend module.
        rt_input (EagerValue): The input ragged tensor, sequence of arrays, or dict.
        default_value (EagerValue): The value used to pad the ragged dimensions.
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Dense padded array.
    """
    if isinstance(rt_input, (list, tuple)) and len(rt_input) > 0 and hasattr(rt_input[0], "shape"):
        max_len = max(x.shape[0] if len(x.shape) > 0 else 1 for x in rt_input)
        pad_val = default_value if default_value is not None else 0.0
        padded = []
        for x in rt_input:
            cur_len = x.shape[0] if len(x.shape) > 0 else 1
            if cur_len < max_len:
                diff = max_len - cur_len
                pad_shape = (diff,) + tuple(x.shape[1:]) if len(x.shape) > 1 else (diff,)
                pad_arr = backend_module.full(pad_shape, pad_val, dtype=getattr(x, "dtype", backend_module.float32))
                padded_x = backend_module.concatenate([x, pad_arr], axis=0)
            else:
                padded_x = x
            padded.append(padded_x)
        return cast(EagerValue, backend_module.stack(padded, axis=0))

    values = None
    row_splits = None
    if isinstance(rt_input, dict) and "values" in rt_input and "row_splits" in rt_input:
        values = rt_input["values"]
        row_splits = rt_input["row_splits"]
    elif hasattr(rt_input, "values") and hasattr(rt_input, "row_splits"):
        values = getattr(rt_input, "values")
        row_splits = getattr(rt_input, "row_splits")

    if values is not None and row_splits is not None:
        num_rows = len(row_splits) - 1
        splits_list = [int(s.item() if hasattr(s, "item") else s) for s in row_splits]
        diffs = [splits_list[i + 1] - splits_list[i] for i in range(num_rows)]
        max_len = max(diffs) if diffs else 0
        pad_val = default_value if default_value is not None else 0.0
        feature_shape = tuple(values.shape[1:]) if len(values.shape) > 1 else ()
        dense_shape = (num_rows, max_len) + feature_shape
        dense_tensor = backend_module.full(dense_shape, pad_val, dtype=getattr(values, "dtype", backend_module.float32))
        rows = []
        for i in range(num_rows):
            start = splits_list[i]
            end = splits_list[i + 1]
            length = end - start
            row_val = values[start:end]
            if length < max_len:
                pad_shape = (max_len - length,) + feature_shape
                pad_arr = backend_module.full(pad_shape, pad_val, dtype=getattr(values, "dtype", backend_module.float32))
                row_val = backend_module.concatenate([row_val, pad_arr], axis=0)
            rows.append(row_val)
        if rows:
            return cast(EagerValue, backend_module.stack(rows, axis=0))
        return cast(EagerValue, dense_tensor)

    return rt_input


@mlx_eager_registry.register("TakeAlongAxis")
def _mlx_take_along_axis(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx take along axis operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.take_along_axis(*args, **kwargs))


@mlx_eager_registry.register("Take")
def _mlx_take(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx take operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.take(*args, **kwargs))


@mlx_eager_registry.register("TensorScatterUpdate")
def _mlx_tensor_scatter_update(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx tensor scatter update operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(cast(EagerTensorProtocol, indices)[..., dim] for dim in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res[idx] = updates
    return cast(EagerValue, res)


@mlx_eager_registry.register("TensorScatterAdd")
def _mlx_tensor_scatter_add(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorScatterAdd.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(cast(EagerTensorProtocol, indices)[..., dim] for dim in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res[idx] = res[idx] + updates
    return cast(EagerValue, res)


@mlx_eager_registry.register("TensorScatterMax")
def _mlx_tensor_scatter_max(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorScatterMax.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    mx = backend_module

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(cast(EagerTensorProtocol, indices)[..., dim] for dim in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res[idx] = mx.maximum(res[idx], updates)
    return cast(EagerValue, res)


@mlx_eager_registry.register("TensorScatterMin")
def _mlx_tensor_scatter_min(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorScatterMin.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    mx = backend_module

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx: tuple[EagerValue, ...] = tuple(cast(EagerTensorProtocol, indices)[..., dim] for dim in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res[idx] = mx.minimum(res[idx], updates)
    return cast(EagerValue, res)


@mlx_eager_registry.register("ScatterNd")
def _mlx_scatter_nd(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx scatter nd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    indices = args[0]
    updates = args[1]
    shape = args[2] if len(args) > 2 else kwargs.get("shape")
    if hasattr(shape, "data"):
        shape = getattr(shape, "data")
    if hasattr(shape, "tolist"):
        shape = getattr(shape, "tolist")()
    if isinstance(shape, tuple):
        shape = list(shape)
    # Implement native MLX ScatterNd logic using array indexing
    res = backend_module.zeros(shape, dtype=getattr(updates, "dtype"))
    idx: tuple[EagerValue, ...] = tuple(cast(EagerTensorProtocol, indices)[..., dim] for dim in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res[idx] = updates
    return cast(EagerValue, res)


@mlx_eager_registry.register("Reshape")
def _mlx_reshape(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx reshape operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else kwargs.get("newshape"))
    if hasattr(shape, "data"):
        shape = getattr(shape, "data")
    if hasattr(shape, "tolist"):
        shape = getattr(shape, "tolist")()
    if isinstance(shape, tuple):
        shape = list(shape)
    return cast(EagerValue, backend_module.reshape(args[0] if "input" not in kwargs else kwargs["input"], shape))


_MLX_DTYPE_FALLBACK_MAP = {
    "bfloat16": "float32",
    "float8": "float32",
    "int4": "int8",
}


def _resolve_dtype(backend_module: ModuleType, dtype_val: EagerValue) -> EagerValue:
    """Resolve a given dtype object or string to a valid MLX dtype.

    Args:
        backend_module (ModuleType): The MLX backend module.
        dtype_val (EagerValue): The requested dtype.

    Returns: mx.array: The resolved MLX dtype.
    """
    if dtype_val is None:
        return None
    dtype_str: str = str(dtype_val).split(".")[-1]
    dtype: EagerValue = getattr(backend_module, dtype_str, dtype_val)
    if isinstance(dtype, str) and not hasattr(backend_module, dtype_str):
        for k, v in _MLX_DTYPE_FALLBACK_MAP.items():
            if k in dtype:
                return cast(EagerValue, getattr(backend_module, v))
    return dtype


@mlx_eager_registry.register("Zeros")
def _mlx_zeros(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx zeros operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = getattr(shape, "data")
    dtype_val = kwargs.get("dtype", None)
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return cast(EagerValue, backend_module.zeros(shape))
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return cast(EagerValue, backend_module.zeros(shape, dtype=dtype))
    except TypeError:
        return cast(EagerValue, backend_module.zeros(shape))


@mlx_eager_registry.register("Ones")
def _mlx_ones(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx ones operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = getattr(shape, "data")
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return cast(EagerValue, backend_module.ones(shape))
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return cast(EagerValue, backend_module.ones(shape, dtype=dtype))
    except TypeError:
        return cast(EagerValue, backend_module.ones(shape))


@mlx_eager_registry.register("Full")
def _mlx_full(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx full operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = getattr(shape, "data")
    fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return cast(EagerValue, backend_module.full(shape, fill_value))
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return cast(EagerValue, backend_module.full(shape, fill_value, dtype=dtype))
    except TypeError:
        return cast(EagerValue, backend_module.full(shape, fill_value))


@mlx_eager_registry.register("Partition")
def _parse_partition_k(k: EagerValue) -> int:
    """Parse k parameter for partition.

    Args:
        k (EagerValue): The k parameter.

    Returns:
        int: Result.
    """
    if hasattr(k, "item"):
        return int(getattr(k, "item")())
    if hasattr(k, "data") and hasattr(getattr(k, "data"), "item"):
        return int(getattr(k, "data").item())
    return int(cast(int, k))


def _mlx_partition(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx partition operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = args[0]
    k_raw = kwargs.get("k", args[1] if len(args) > 1 else 1)
    k: int = int(cast(int, _parse_partition_k(k_raw)))
    return_indices = kwargs.get("return_indices", None)
    kth: int = max(0, getattr(a, "shape")[-1] - k)
    if return_indices is False:
        if hasattr(backend_module, "topk"):
            return cast(EagerValue, backend_module.topk(a, k))
        return cast(EagerValue, backend_module.partition(a, kth, axis=-1)[..., -k:])
    indices = backend_module.argpartition(a, kth, axis=-1)[..., -k:]
    if return_indices is True:
        return cast(EagerValue, indices)
    values = backend_module.take_along_axis(a, indices, axis=-1)
    return (values, indices)


@mlx_eager_registry.register("NanToNum")
def _mlx_nan_to_num(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx nan to num operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    valid_kwargs: dict[str, float] = {}
    for key in ("nan", "posinf", "neginf"):
        if key in kwargs:
            val: float = cast(float, kwargs[key])
            if hasattr(val, "item"):
                val = float(val.item())
            elif hasattr(val, "data") and hasattr(val.data, "item"):
                val = float(val.data.item())
            elif val is not None:
                val = float(val)
            valid_kwargs[key] = val
    return cast(EagerValue, backend_module.nan_to_num(*args, **valid_kwargs))


@mlx_eager_registry.register("Cummax")
def _mlx_cummax(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx cummax operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: str = cast(str, kwargs.pop("dtype", None))
    res: str = backend_module.cummax(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = getattr(res, "astype")(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cummin")
def _mlx_cummin(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx cummin operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: str = cast(str, kwargs.pop("dtype", None))
    res: str = backend_module.cummin(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = getattr(res, "astype")(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cumprod")
def _mlx_cumprod(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx cumprod operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: str = cast(str, kwargs.pop("dtype", None))
    res: str = backend_module.cumprod(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = getattr(res, "astype")(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Slice")
def _mlx_slice(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx slice operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = args[0]
    dim: int = int(cast(int, kwargs.get("dim", 0)))
    start = cast("int | None", kwargs.get("start"))
    end = cast("int | None", kwargs.get("end"))
    step = cast("int | None", kwargs.get("step", 1))
    sl: list[slice] = [builtins.slice(None)] * len(getattr(a, "shape"))
    sl[dim] = builtins.slice(start, end, step)
    return cast(EagerValue, getattr(a, "__getitem__")(tuple(sl)))


@mlx_eager_registry.register("Eye")
def _mlx_eye(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx eye operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    n_arg = args[0]
    if hasattr(n_arg, "data"):
        n_arg = getattr(n_arg, "data")
    m_arg = args[1] if len(args) > 1 else None
    if hasattr(m_arg, "data"):
        m_arg = getattr(m_arg, "data")
    m_val: float = int(cast(int, m_arg)) if m_arg is not None else int(cast(int, n_arg))
    k_val: float = int(cast(int, kwargs.get("k", 0)))
    return cast(EagerValue, getattr(backend_module, "eye")(n=int(cast(int, n_arg)), m=m_val, k=k_val, dtype=getattr(backend_module, getattr(kwargs, "get")("dtype", "float32"))))


@mlx_eager_registry.register("Rope")
def _mlx_rope(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Apply Rotary Positional Encoding using MLX.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dim = kwargs.get("dim")
    base: float = cast(float, kwargs.get("base", 10000.0))
    offset: int = cast(int, kwargs.get("offset", 0))
    traditional: bool = cast(bool, kwargs.get("traditional", False))
    scale: float = cast(float, kwargs.get("scale", 1.0))
    return cast(EagerValue, getattr(mx.fast, "rope")(x, dim, traditional=traditional, base=base, scale=scale, offset=offset))


@mlx_eager_registry.register("Variance")
def _mlx_variance(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mlx variance operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs.setdefault("ddof", 0)
    return cast(EagerValue, backend_module.var(*args, **kwargs))


@mlx_eager_registry.register("AllReduce")
def _mlx_all_reduce(backend_module: ModuleType, tensor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement AllReduce for MLX eager mode.

    Args:
        backend_module (ModuleType): The MLX backend module.
        tensor (EagerValue): The input tensor.
        **kwargs (EagerValue): Keyword args.

    Returns: mx.array: The reduced tensor.
    """
    mx = backend_module

    if hasattr(mx, "distributed") and hasattr(mx.distributed, "all_sum"):
        return cast(EagerValue, mx.distributed.all_sum(tensor))
    return tensor


@mlx_eager_registry.register("AllGather")
def _mlx_all_gather(backend_module: ModuleType, tensor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement AllGather for MLX eager mode.

    Args:
        backend_module (ModuleType): The MLX backend module.
        tensor (EagerValue): The input tensor.
        **kwargs (EagerValue): Keyword args.

    Returns: mx.array: The gathered tensor.
    """
    mx = backend_module

    if hasattr(mx, "distributed") and hasattr(mx.distributed, "all_gather"):
        return cast(EagerValue, mx.distributed.all_gather(tensor))
    return cast(EagerValue, mx.expand_dims(tensor, axis=0))


@mlx_eager_registry.register("AllToAll")
def _mlx_all_to_all(backend_module: ModuleType, tensor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement AllToAll for MLX eager mode.

    Args:
        backend_module (ModuleType): The MLX backend module.
        tensor (EagerValue): The input tensor.
        **kwargs (EagerValue): Keyword args.

    Returns: mx.array: The distributed tensor.
    """
    mx = backend_module

    if hasattr(mx, "distributed") and hasattr(mx.distributed, "all_to_all"):
        return cast(EagerValue, mx.distributed.all_to_all(tensor))
    return tensor


def _slice_scatter_chunk(tensor: EagerValue, scatter_dim: int, rank: int, world_size: int) -> EagerValue:
    """Slice a tensor along scatter_dim for a given rank and world_size.

    Args:
        tensor (EagerValue): Input tensor to slice.
        scatter_dim (int): Dimension along which to partition.
        rank (int): Current rank index.
        world_size (int): Total number of ranks.

    Returns:
        EagerValue: Sliced chunk for the specified rank.
    """
    t_arr = cast(EagerTensorProtocol, tensor)
    dim_size = t_arr.shape[scatter_dim]
    chunk_size = dim_size // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    slices = [slice(None)] * t_arr.ndim
    slices[scatter_dim] = slice(start_idx, end_idx)
    return t_arr[tuple(slices)]


def _mlx_native_reduce_scatter(mx_mod: EagerValue, tensor: EagerValue, kwargs: dict[str, EagerValue]) -> EagerValue:
    """Execute native MLX distributed all_sum followed by rank chunk slicing.

    Args:
        mx_mod (EagerValue): Loaded MLX module with distributed support.
        tensor (EagerValue): Input tensor.
        kwargs (dict[str, EagerValue]): Distributed options (group, scatter_dim).

    Returns:
        EagerValue: Reduced and scattered tensor slice.
    """
    mod = cast(ModuleType, mx_mod)
    reduced = mod.distributed.all_sum(tensor)
    group = cast(ModuleType, kwargs.get("group"))
    try:
        rank = group.rank() if group else (mod.distributed.init().rank() if hasattr(mod.distributed, "init") else 0)
        world_size = group.size() if group else (mod.distributed.init().size() if hasattr(mod.distributed, "init") else 1)
    except Exception:
        rank = 0
        world_size = 1

    scatter_dim = int(cast(int, kwargs.get("scatter_dim", 0)))
    return _slice_scatter_chunk(reduced, scatter_dim, rank, world_size)


def _combine_ring_chunks(mx_mod: EagerValue, accum: EagerValue, peer_chunk: EagerValue, red_op: str) -> EagerValue:
    """Combine two tensor chunks using the specified reduction operator.

    Args:
        mx_mod (EagerValue): Loaded MLX module or None.
        accum (EagerValue): Running accumulator chunk.
        peer_chunk (EagerValue): Incoming peer chunk.
        red_op (str): Uppercase reduction operation name.

    Returns:
        EagerValue: Combined chunk result.
    """
    mod = cast(ModuleType, mx_mod)
    acc_arr = cast(EagerTensorProtocol, accum)
    peer_arr = cast(EagerTensorProtocol, peer_chunk)
    if red_op in ("PROD", "PRODUCT", "MUL"):
        return mod.multiply(acc_arr, peer_arr) if (mod and hasattr(mod, "multiply")) else (acc_arr * peer_arr)
    if red_op == "MAX":
        return mod.maximum(acc_arr, peer_arr) if (mod and hasattr(mod, "maximum")) else (acc_arr if bool(acc_arr >= peer_arr) else peer_arr)
    if red_op == "MIN":
        return mod.minimum(acc_arr, peer_arr) if (mod and hasattr(mod, "minimum")) else (acc_arr if bool(acc_arr <= peer_arr) else peer_arr)
    return mod.add(acc_arr, peer_arr) if (mod and hasattr(mod, "add")) else (acc_arr + peer_arr)


def _mlx_ring_reduce_scatter(
    mx_mod: EagerValue,
    all_ranks_data: list[EagerValue] | tuple[EagerValue, ...],
    scatter_dim: int,
    rank: int,
    red_op: str,
) -> EagerValue:
    """Perform host-level ring reduce-scatter across all_ranks_data.

    Args:
        mx_mod (EagerValue): Loaded MLX module or None.
        all_ranks_data (list[EagerValue] | tuple[EagerValue, ...]): Per-rank input tensors.
        scatter_dim (int): Dimension to scatter along.
        rank (int): Current rank index.
        red_op (str): Uppercase reduction operator string.

    Returns:
        EagerValue: Accumulated chunk for the current rank.
    """
    num_ranks = len(all_ranks_data)
    dim_size = getattr(all_ranks_data[0], "shape", [1])[scatter_dim]
    chunk_size = dim_size // num_ranks
    start = rank * chunk_size
    end = start + chunk_size
    slices = [slice(None)] * getattr(all_ranks_data[0], "ndim", 1)
    slices[scatter_dim] = slice(start, end)

    accum: EagerValue = cast(EagerTensorProtocol, all_ranks_data[rank])[tuple(slices)]
    for step in range(1, num_ranks):
        peer = (rank + step) % num_ranks
        peer_chunk = cast(EagerTensorProtocol, all_ranks_data[peer])[tuple(slices)]
        accum = _combine_ring_chunks(mx_mod, accum, peer_chunk, red_op)
    return accum


@mlx_eager_registry.register("ReduceScatter")
def _mlx_reduce_scatter(backend_module: ModuleType, tensor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ReduceScatter for MLX eager mode using native collectives or ring reduce-scatter fallback.

    Args:
        backend_module (ModuleType): The MLX backend module.
        tensor (EagerValue): The input tensor.
        **kwargs (EagerValue): Keyword args including op_type, scatter_dim, rank, world_size, all_ranks_data.

    Returns:
        EagerValue: The scattered reduced tensor.
    """
    mx_mod = backend_module

    if mx_mod is not None and hasattr(mx_mod, "distributed") and hasattr(mx_mod.distributed, "all_sum"):
        return _mlx_native_reduce_scatter(mx_mod, tensor, kwargs)

    all_ranks_data = kwargs.get("all_ranks_data")
    scatter_dim = int(cast(int, kwargs.get("scatter_dim", 0)))
    red_op = str(kwargs.get("op_type", "SUM")).upper()
    rank = int(cast(int, kwargs.get("rank", 0)))

    if all_ranks_data:
        return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _mlx_ring_reduce_scatter)(mx_mod, all_ranks_data, scatter_dim, rank, red_op))

    world_size = int(cast(int, kwargs.get("world_size", 1)))
    if hasattr(tensor, "shape"):
        if world_size <= 1:
            return tensor
        return _slice_scatter_chunk(tensor, scatter_dim, rank, world_size)

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raise BackendNotSupportedError("ReduceScatter is not supported in the current MLX version.")
