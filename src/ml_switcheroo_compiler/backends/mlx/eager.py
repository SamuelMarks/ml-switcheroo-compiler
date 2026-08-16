"""Module eager.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""
import builtins
from typing import Any

import mlx.core as mx

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, mlx_eager_registry


def _get_mlx_func(op_type: str) -> Any:
    """Retrieve the corresponding MLX function for a given operation type.

    Args:
        op_type (str): The name of the operation.

    Returns: Any: The MLX function if found, otherwise None.
    """
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    if snake == "mul":
        snake = "multiply"
    elif snake == "sub":
        snake = "subtract"
    elif snake == "div":
        snake = "divide"
    for mod in [mx, getattr(mx, "linalg", None), getattr(mx, "fft", None)]:
        if mod is not None and hasattr(mod, snake):
            return getattr(mod, snake)
    return None


# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Evaluate execute_op operation.

    Args:
        cls (type): The class.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

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
        return func(*args, **kwargs)
    except AttributeError:
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        raise BackendNotSupportedError("Operation " + op_type + " is not implemented for the MLX backend.") from None


@mlx_eager_registry.register("Cast")
def _mlx_cast(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_cast operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    tensor = args[0]
    dtype_val = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return tensor
    return tensor.astype(dtype)


@mlx_eager_registry.register("RaggedTensorToDense")
def _mlx_ragged_tensor_to_dense(backend_module: Any, rt_input: Any, **kwargs: Any) -> Any:
    """Convert a ragged tensor to dense using MLX (stubbed).

    Args:
        backend_module (object): The MLX backend module.
        rt_input (object): The input ragged tensor.
        **kwargs (object): Keyword arguments.

    Returns: Any: The input tensor unchanged.
    """
    return rt_input


@mlx_eager_registry.register("TakeAlongAxis")
def _mlx_take_along_axis(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_take_along_axis operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.take_along_axis(*args, **kwargs)


@mlx_eager_registry.register("Take")
def _mlx_take(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_take operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.take(*args, **kwargs)


@mlx_eager_registry.register("TensorScatterUpdate")
def _mlx_tensor_scatter_update(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_tensor_scatter_update operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = updates
    return res


@mlx_eager_registry.register("TensorScatterAdd")
def _mlx_tensor_scatter_add(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorScatterAdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = res[idx] + updates
    return res


@mlx_eager_registry.register("TensorScatterMax")
def _mlx_tensor_scatter_max(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorScatterMax.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import mlx.core as mx

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = mx.maximum(res[idx], updates)
    return res


@mlx_eager_registry.register("TensorScatterMin")
def _mlx_tensor_scatter_min(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorScatterMin.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import mlx.core as mx

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = mx.minimum(res[idx], updates)
    return res


@mlx_eager_registry.register("ScatterNd")
def _mlx_scatter_nd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_scatter_nd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    indices = args[0]
    updates = args[1]
    shape = args[2] if len(args) > 2 else kwargs.get("shape")
    if hasattr(shape, "data"):
        shape = shape.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if hasattr(shape, "tolist"):
        shape = shape.tolist()  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if isinstance(shape, tuple):
        shape = list(shape)
    # Implement native MLX ScatterNd logic using array indexing
    res = backend_module.zeros(shape, dtype=updates.dtype)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = updates
    return res


@mlx_eager_registry.register("Reshape")
def _mlx_reshape(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_reshape operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else kwargs.get("newshape"))
    if hasattr(shape, "data"):
        shape = shape.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if hasattr(shape, "tolist"):
        shape = shape.tolist()  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if isinstance(shape, tuple):
        shape = list(shape)
    return backend_module.reshape(args[0] if "input" not in kwargs else kwargs["input"], shape)


_MLX_DTYPE_FALLBACK_MAP = {
    "bfloat16": "float32",
    "float8": "float32",
    "int4": "int8",
}


def _resolve_dtype(backend_module: Any, dtype_val: Any) -> Any:
    """Resolve a given dtype object or string to a valid MLX dtype.

    Args:
        backend_module (object): The MLX backend module.
        dtype_val (object): The requested dtype.

    Returns: Any: The resolved MLX dtype.
    """
    if dtype_val is None:
        return None
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    if isinstance(dtype, str) and not hasattr(backend_module, dtype_str):
        for k, v in _MLX_DTYPE_FALLBACK_MAP.items():
            if k in dtype:
                return getattr(backend_module, v)
    return dtype


@mlx_eager_registry.register("Zeros")
def _mlx_zeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_zeros operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", None)
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return backend_module.zeros(shape)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return backend_module.zeros(shape, dtype=dtype)
    except TypeError:
        return backend_module.zeros(shape)


@mlx_eager_registry.register("Ones")
def _mlx_ones(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_ones operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return backend_module.ones(shape)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return backend_module.ones(shape, dtype=dtype)
    except TypeError:
        return backend_module.ones(shape)


@mlx_eager_registry.register("Full")
def _mlx_full(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_full operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return backend_module.full(shape, fill_value)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return backend_module.full(shape, fill_value, dtype=dtype)
    except TypeError:
        return backend_module.full(shape, fill_value)


@mlx_eager_registry.register("Partition")
def _parse_partition_k(k: Any) -> int:
    """Parse k parameter for partition.

    Args:
        k (object): The k parameter.

    Returns:
        int: Result.
    """
    if hasattr(k, "item"):
        return int(k.item())
    if hasattr(k, "data") and hasattr(k.data, "item"):
        return int(k.data.item())
    return int(k)


def _mlx_partition(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_partition operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    k_raw = kwargs.get("k", args[1] if len(args) > 1 else 1)
    k = _parse_partition_k(k_raw)
    return_indices = kwargs.get("return_indices", None)
    kth = max(0, a.shape[-1] - k)
    if return_indices is False:
        if hasattr(backend_module, "topk"):
            return backend_module.topk(a, k)
        return backend_module.partition(a, kth, axis=-1)[..., -k:]
    indices = backend_module.argpartition(a, kth, axis=-1)[..., -k:]
    if return_indices is True:
        return indices
    values = backend_module.take_along_axis(a, indices, axis=-1)
    return (values, indices)


@mlx_eager_registry.register("NanToNum")
def _mlx_nan_to_num(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_nan_to_num operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    valid_kwargs = {}
    for key in ("nan", "posinf", "neginf"):
        if key in kwargs:
            val = kwargs[key]
            if hasattr(val, "item"):
                val = float(val.item())
            elif hasattr(val, "data") and hasattr(val.data, "item"):
                val = float(val.data.item())
            elif val is not None:
                val = float(val)
            valid_kwargs[key] = val
    return backend_module.nan_to_num(*args, **valid_kwargs)


@mlx_eager_registry.register("Cummax")
def _mlx_cummax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_cummax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cummax(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cummin")
def _mlx_cummin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_cummin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cummin(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cumprod")
def _mlx_cumprod(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_cumprod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cumprod(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Slice")
def _mlx_slice(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    dim = kwargs.get("dim")
    start = kwargs.get("start")
    end = kwargs.get("end")
    step = kwargs.get("step", 1)
    sl = [builtins.slice(None)] * len(a.shape)
    sl[dim] = builtins.slice(start, end, step)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return a[tuple(sl)]


@mlx_eager_registry.register("Eye")
def _mlx_eye(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_eye operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    n_arg = args[0]
    if hasattr(n_arg, "data"):
        n_arg = n_arg.data
    m_arg = args[1] if len(args) > 1 else None
    if hasattr(m_arg, "data"):
        m_arg = m_arg.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    m_val = int(m_arg) if m_arg is not None else int(n_arg)
    k_val = int(kwargs.get("k", 0))
    return backend_module.eye(n=int(n_arg), m=m_val, k=k_val, dtype=getattr(backend_module, kwargs.get("dtype", "float32")))


@mlx_eager_registry.register("Rope")
def _mlx_rope(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Apply Rotary Positional Encoding using MLX.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dim = kwargs.get("dim")
    base = kwargs.get("base", 10000.0)
    offset = kwargs.get("offset", 0)
    traditional = kwargs.get("traditional", False)
    scale = kwargs.get("scale", 1.0)
    return mx.fast.rope(x, dim, traditional=traditional, base=base, scale=scale, offset=offset)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


@mlx_eager_registry.register("Variance")
def _mlx_variance(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mlx_variance operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    kwargs.setdefault("ddof", 0)
    return backend_module.var(*args, **kwargs)
