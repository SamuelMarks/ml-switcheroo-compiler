# ruff: noqa: E501
"""Backend utilities."""

import builtins

import mlx.core as mx

from ml_switcheroo_compiler.backends.eager_registry import mlx_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import execute_op as np_execute_op


def _to_numpy(val: object) -> object:
    """Evaluate and process the to numpy operation.

    Args:
        val (object): Required parameter for val.

    Returns:
        object: The evaluated or processed output.
    """
    import numpy as np

    return np.array(val)


def _from_numpy(val: object) -> object:
    """From numpy."""
    import mlx.core as mx

    return mx.array(val)


def _execute_numpy_fallback(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Evaluate and process the execute numpy fallback operation.

    Args:
        cls (type): Required parameter for cls.
        op_type (str): Required parameter for op_type.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    np_args = [_to_numpy(a) for a in args]
    np_kwargs = {k: _to_numpy(v) for (k, v) in kwargs.items()}
    res = np_execute_op(cls, op_type, *np_args, **np_kwargs)
    return _from_numpy(res)


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The cls parameter for the operation.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    try:
        if "dim" in kwargs and op_type not in ("TakeAlongAxis", "Take"):
            kwargs["axis"] = kwargs.pop("dim")
        func_registry = mlx_eager_registry.get(op_type)
        if func_registry is not None:
            return func_registry(mx, *args, **kwargs)
        return _execute_numpy_fallback(cls, op_type, *args, **kwargs)
    except AttributeError:
        return _execute_numpy_fallback(cls, op_type, *args, **kwargs)


@mlx_eager_registry.register("Cast")
def _mlx_cast(backend_module: object, *args: object, **kwargs: object) -> object:
    tensor = args[0]
    dtype_val = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return tensor
    return tensor.astype(dtype)


@mlx_eager_registry.register("RaggedTensorToDense")
def _mlx_ragged_tensor_to_dense(backend_module: object, rt_input: object, **kwargs: object) -> object:
    return rt_input


@mlx_eager_registry.register("TakeAlongAxis")
def _mlx_take_along_axis(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx take along axis operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.take_along_axis(*args, **kwargs)


@mlx_eager_registry.register("Take")
def _mlx_take(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx take operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.take(*args, **kwargs)


@mlx_eager_registry.register("TensorScatterUpdate")
def _mlx_tensor_scatter_update(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx tensor scatter update operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = updates
    return res


@mlx_eager_registry.register("TensorScatterAdd")
def _mlx_tensor_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement TensorScatterAdd."""
    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = res[idx] + updates
    return res


@mlx_eager_registry.register("TensorScatterMax")
def _mlx_tensor_scatter_max(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement TensorScatterMax."""
    import mlx.core as mx

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = mx.maximum(res[idx], updates)
    return res


@mlx_eager_registry.register("TensorScatterMin")
def _mlx_tensor_scatter_min(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement TensorScatterMin."""
    import mlx.core as mx

    (tensor, indices, updates) = (args[0], args[1], args[2])
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = mx.minimum(res[idx], updates)
    return res


@mlx_eager_registry.register("ScatterNd")
def _mlx_scatter_nd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx scatter nd operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    # indices = args[0]
    # updates = args[1]
    shape = args[2] if len(args) > 2 else kwargs.get("shape")
    if hasattr(shape, "data"):
        shape = shape.data
    if hasattr(shape, "tolist"):
        shape = shape.tolist()
    if isinstance(shape, tuple):
        shape = list(shape)

    # Simple fallback using NumPy since MLX scatter_nd can be tricky with exact API.
    return _execute_numpy_fallback(None, "ScatterNd", *args, **kwargs)


@mlx_eager_registry.register("Reshape")
def _mlx_reshape(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx reshape operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else kwargs.get("newshape"))
    if hasattr(shape, "data"):
        shape = shape.data
    if hasattr(shape, "tolist"):
        shape = shape.tolist()
    if isinstance(shape, tuple):
        shape = list(shape)
    return backend_module.reshape(args[0] if "input" not in kwargs else kwargs["input"], shape)


_MLX_DTYPE_FALLBACK_MAP = {
    "bfloat16": "float32",
    "float8": "float32",
    "int4": "int8",
}


def _resolve_dtype(backend_module: object, dtype_val: object) -> object:
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
def _mlx_zeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx zeros operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _mlx_ones(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx ones operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _mlx_full(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx full operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _parse_partition_k(k: object) -> int:
    """Parse k parameter for partition."""
    if hasattr(k, "item"):
        return int(k.item())
    if hasattr(k, "data") and hasattr(k.data, "item"):
        return int(k.data.item())
    return int(k)


def _mlx_partition(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx partition operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _mlx_nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx nan to num operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _mlx_cummax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx cummax operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cummax(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cummin")
def _mlx_cummin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx cummin operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cummin(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Cumprod")
def _mlx_cumprod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx cumprod operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.cumprod(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        res = res.astype(getattr(backend_module, str(getattr(dtype, "value", dtype)), getattr(dtype, "value", dtype)))
    return res


@mlx_eager_registry.register("Slice")
def _mlx_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx slice operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    a = args[0]
    dim = kwargs.get("dim")
    start = kwargs.get("start")
    end = kwargs.get("end")
    step = kwargs.get("step", 1)
    sl = [builtins.slice(None)] * len(a.shape)
    sl[dim] = builtins.slice(start, end, step)
    return a[tuple(sl)]


@mlx_eager_registry.register("Eye")
def _mlx_eye(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx eye operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    n_arg = args[0]
    if hasattr(n_arg, "data"):
        n_arg = n_arg.data
    m_arg = args[1] if len(args) > 1 else None
    if hasattr(m_arg, "data"):
        m_arg = m_arg.data
    m_val = int(m_arg) if m_arg is not None else int(n_arg)
    k_val = int(kwargs.get("k", 0))
    return backend_module.eye(n=int(n_arg), m=m_val, k=k_val, dtype=getattr(backend_module, kwargs.get("dtype", "float32")))


@mlx_eager_registry.register("Rope")
def _mlx_rope(backend_module: object, x: object, **kwargs: object) -> object:
    """Apply Rotary Positional Encoding using MLX.

    Args:
        backend_module: Arg.
        x: Arg.
        kwargs: Arg.
    """
    dim = kwargs.get("dim")
    base = kwargs.get("base", 10000.0)
    offset = kwargs.get("offset", 0)
    traditional = kwargs.get("traditional", False)
    scale = kwargs.get("scale", 1.0)
    return mx.fast.rope(x, dim, traditional=traditional, base=base, scale=scale, offset=offset)


@mlx_eager_registry.register("Variance")
def _mlx_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the mlx variance operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    kwargs.setdefault("ddof", 0)
    return backend_module.var(*args, **kwargs)
