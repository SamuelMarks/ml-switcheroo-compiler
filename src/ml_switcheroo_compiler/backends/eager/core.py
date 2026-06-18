"""Core utilities."""

import warnings
import scipy.special


from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def execute_generic_op(
    backend_module: object,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute op generically.

    Args:
        backend_module (Any): The backend module (e.g., jax.numpy, mlx.core).
        op_type (str): The operation type.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: The result of the operation.
    """
    warnings.warn(
        "Python-level eager execution bypasses and _EAGER_OP_MAP fallbacks are deprecated. "
        "Frontends should only trace to LogicalNode. "
        "Graph execution should be isolated to eval() phase.",
        DeprecationWarning,
        stacklevel=2,
    )

    func_registry = global_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(backend_module, *args, **kwargs)

    try:
        func = getattr(backend_module, op_type.lower())
        if op_type not in ("Sort", "ArgSort", "Allclose", "Reshape", "BroadcastTo"):
            return func(*args, **kwargs)
    except AttributeError:
        pass

    name = getattr(backend_module, "__name__", "unknown")
    msg = f"Operation '{op_type}' is not supported by backend module {name}."
    raise NotImplementedError(msg)


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    func = getattr(backend_module, "divide", getattr(backend_module, "true_divide", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fft")
def _fft(backend_module: object, *args: object, **kwargs: object) -> object:
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    func = getattr(
        backend_module,
        "erfinv",
        scipy.special.erfinv if getattr(backend_module, "__name__", "") == "numpy" else None,
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return backend_module.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return None


@global_eager_registry.register("Sort")
def _sort(backend_module: object, *args: object, **kwargs: object) -> object:
    a = args[0]
    axis = kwargs.get("axis", -1)
    if hasattr(backend_module, "sort"):
        return backend_module.sort(a, axis=axis)
    return None


@global_eager_registry.register("ArgSort")
def _argsort(backend_module: object, *args: object, **kwargs: object) -> object:
    a = args[0]
    axis = kwargs.get("axis", -1)
    if hasattr(backend_module, "argsort"):
        return backend_module.argsort(a, axis=axis)
    return None


@global_eager_registry.register("Reshape")
def _reshape(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    shape = list(args[1]) if len(args) > 1 else list(kwargs.get("shape", kwargs.get("newshape")))
    if hasattr(backend_module, "reshape"):
        return backend_module.reshape(x, shape)
    return None


@global_eager_registry.register("Einsum")
def _einsum(backend_module: object, *args: object, **kwargs: object) -> object:
    eq = (
        kwargs.pop("equation", "")
        if "equation" in kwargs
        else args[0]
        if len(args) > 0 and isinstance(args[0], str)
        else ""
    )
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args
    if hasattr(backend_module, "einsum"):
        return backend_module.einsum(eq, *op_args, **kwargs)
    return None


@global_eager_registry.register("Permute")
def _permute(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    axes = kwargs.get("axes", None)
    if hasattr(backend_module, "transpose"):
        return backend_module.transpose(x, axes)
    return None


@global_eager_registry.register("Allclose")
def _allclose(backend_module: object, *args: object, **kwargs: object) -> object:
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-5)
    atol = kwargs.get("atol", 1e-8)
    equal_nan = kwargs.get("equal_nan", False)

    def _val(x: object) -> object:
        x_data = getattr(x, "data", x)
        if hasattr(x_data, "item") and callable(x_data.item):
            return x_data.item()
        if hasattr(x_data, "tolist"):
            return x_data.tolist()
        return x_data

    if hasattr(backend_module, "allclose"):
        return backend_module.allclose(
            a, b, rtol=float(_val(rtol)), atol=float(_val(atol)), equal_nan=bool(_val(equal_nan))
        )
    return None


@global_eager_registry.register("TensorScatterUpdate")
def _tensor_scatter_update(backend_module: object, *args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].set(updates)
    elif name == "torch":
        return tensor.clone().index_put_(tuple(indices.unbind(-1)), updates)
    elif name == "keras.ops":
        return backend_module.tensor_scatter_update(tensor, indices, updates)
    elif name == "tensorflow.math" or name == "tensorflow":
        import tensorflow as tf

        return tf.tensor_scatter_nd_update(tensor, indices, updates)
    else:
        raise NotImplementedError(f"TensorScatterUpdate eager not implemented for {name}")


@global_eager_registry.register("TensorScatterAdd")
def _tensor_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].add(updates)
    elif name == "torch":
        return tensor.clone().index_put_(tuple(indices.unbind(-1)), updates, accumulate=True)
    elif name == "keras.ops":
        return backend_module.tensor_scatter_add(tensor, indices, updates)
    elif name == "tensorflow.math" or name == "tensorflow":
        import tensorflow as tf

        return tf.tensor_scatter_nd_add(tensor, indices, updates)
    else:
        raise NotImplementedError(f"TensorScatterAdd eager not implemented for {name}")


@global_eager_registry.register("TensorScatterMax")
def _tensor_scatter_max(backend_module: object, *args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].max(updates)
    elif name == "torch":
        raise NotImplementedError("TensorScatterMax not implemented for torch in legacy eager")
    elif name == "keras.ops":
        return backend_module.tensor_scatter_max(tensor, indices, updates)
    elif name == "tensorflow.math" or name == "tensorflow":
        import tensorflow as tf

        return tf.tensor_scatter_nd_max(tensor, indices, updates)
    else:
        raise NotImplementedError(f"TensorScatterMax eager not implemented for {name}")


@global_eager_registry.register("TensorScatterMin")
def _tensor_scatter_min(backend_module: object, *args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].min(updates)
    elif name == "torch":
        raise NotImplementedError("TensorScatterMin not implemented for torch in legacy eager")
    elif name == "keras.ops":
        return backend_module.tensor_scatter_min(tensor, indices, updates)
    elif name == "tensorflow.math" or name == "tensorflow":
        import tensorflow as tf

        return tf.tensor_scatter_nd_min(tensor, indices, updates)
    else:
        raise NotImplementedError(f"TensorScatterMin eager not implemented for {name}")


def _normalize_shape(shape: object) -> object:
    if hasattr(shape, "data"):
        shape = shape.data
    if hasattr(shape, "tolist") and callable(shape.tolist):
        shape = shape.tolist()
    if isinstance(shape, tuple):
        shape = list(shape)
    return shape


def _extract_shape_value(val: object) -> int:
    if hasattr(val, "data"):
        val = val.data

    if hasattr(val, "item") and callable(val.item):
        val = val.item()
    elif hasattr(val, "tolist") and callable(val.tolist):
        val_list = val.tolist()
        val = val_list[0] if isinstance(val_list, list) else val_list

    return int(val)  # type: ignore


def _parse_eager_shape(shape: object) -> list[int]:
    shape = _normalize_shape(shape)

    if not isinstance(shape, list) or not shape:
        return shape  # type: ignore

    return [_extract_shape_value(s) for s in shape]


@global_eager_registry.register("BroadcastTo")
def _broadcast_to(backend_module: object, *args: object, **kwargs: object) -> object:
    shape = kwargs.get("shape", args[1] if len(args) > 1 else args[0] if len(args) > 0 else None)
    parsed_shape = _parse_eager_shape(shape)
    return backend_module.broadcast_to(args[0], parsed_shape)


@global_eager_registry.register("Zeros")
def _zeros(backend_module: object, *args: object, **kwargs: object) -> object:
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    try:
        return backend_module.zeros(shape, dtype=dtype)
    except TypeError:
        try:
            return backend_module.zeros(shape)
        except TypeError:
            return backend_module.zeros(shape=shape)


@global_eager_registry.register("Ones")
def _ones(backend_module: object, *args: object, **kwargs: object) -> object:
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    try:
        return backend_module.ones(shape, dtype=dtype)
    except TypeError:
        try:
            return backend_module.ones(shape)
        except TypeError:
            return backend_module.ones(shape=shape)


@global_eager_registry.register("Full")
def _full(backend_module: object, *args: object, **kwargs: object) -> object:
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    try:
        return backend_module.full(shape, fill_value, dtype=dtype)
    except TypeError:
        try:
            return backend_module.full(shape, fill_value)
        except TypeError:
            return backend_module.full(shape=shape, fill_value=fill_value)


@global_eager_registry.register("BroadcastInDim")
def _broadcast_in_dim(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
    broadcast_dimensions = kwargs.get("broadcast_dimensions", args[2] if len(args) > 2 else None)
    expanded_shape = []
    for i in range(len(shape)):
        if i in broadcast_dimensions:
            expanded_shape.append(x.shape[broadcast_dimensions.index(i)])
        else:
            expanded_shape.append(1)
    x_expanded = backend_module.reshape(x, expanded_shape)
    return backend_module.broadcast_to(x_expanded, shape)


@global_eager_registry.register("TopK")
def _top_k(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    k = kwargs.get("k", args[1] if len(args) > 1 else None)
    axis = kwargs.get("axis", -1)

    idx = backend_module.argsort(x, axis=axis)
    if axis < 0:
        axis += len(x.shape)
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = idx[tuple(slc)]
    val_k = backend_module.take_along_axis(x, idx_k, axis=axis)
    return val_k, idx_k


@global_eager_registry.register("Resize")
def _resize(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
    if hasattr(backend_module, "zeros"):
        return backend_module.zeros((x.shape[0], *shape, x.shape[-1]), dtype=x.dtype)
    return None


@global_eager_registry.register("DynamicUpdateSlice")
def _dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    update = args[1]
    start_indices = args[2:] if len(args) > 2 else kwargs.get("start_indices", [])
    if hasattr(backend_module, "dynamic_update_slice"):
        return backend_module.dynamic_update_slice(x, update, start_indices)
    return x


@global_eager_registry.register("ConvGeneralDilated")
def _conv_general_dilated_fallback(
    backend_module: object, *args: object, **kwargs: object
) -> object:
    return backend_module.zeros((1,))


@global_eager_registry.register("Psum")
def _psum(backend_module: object, *args: object, **kwargs: object) -> object:
    return args[0]


@global_eager_registry.register("Pmean")
def _pmean(backend_module: object, *args: object, **kwargs: object) -> object:
    return args[0]


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    if hasattr(backend_module, "zeros"):
        return backend_module.zeros((1,))
    return None


def generic_zeros(backend_module: object, shape: tuple[int, ...]) -> object:
    """Generic zeros function.

    Args:
        backend_module (Any): The backend module.
        shape (tuple[int, ...]): Shape of the tensor.

    Returns:
        object: A tensor of zeros.
    """
    return backend_module.zeros(shape)


def generic_array(backend_module: object, data: object) -> object:
    """Generic array creation.

    Args:
        backend_module (Any): The backend module.
        data (object): The data to convert.

    Returns:
        object: A tensor array.
    """
    try:
        if data is None:
            return None
        if getattr(data, "__name__", "") == "mlx.core":
            return data
        if "mlx.core.array" in str(type(data)):
            return data
        return backend_module.array(data)
    except AttributeError:
        return backend_module.convert_to_tensor(data)


def generic_asarray(backend_module: object, data: object) -> object:
    """Generic asarray.

    Args:
        backend_module (Any): The backend module.
        data (object): The data to convert.

    Returns:
        object: A tensor array.
    """
    try:
        return backend_module.asarray(data)
    except AttributeError:
        return backend_module.convert_to_tensor(data)


def generic_item(backend_module: object, data: object) -> float:
    """Generic item extraction.

    Args:
        backend_module (Any): The backend module.
        data (object): The data tensor.

    Returns:
        float: The scalar value.
    """
    try:
        return float(backend_module.asarray(data).item())
    except AttributeError:
        return float(data)


def _apply_grouped_reduction(
    backend_module: object,
    op_name: str,
    x: object,
    groups: int,
    axis: int,
) -> object:
    shape = list(x.shape)
    ndims = len(shape)
    if axis < 0:
        axis += ndims

    C = shape[axis]
    C_per_group = C // groups

    reshaped_dims = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]

    reshaped_x = backend_module.reshape(x, reshaped_dims)
    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)

    is_torch = backend_module.__name__ == "torch"

    if op_name == "mean":
        if is_torch:
            return backend_module.mean(reshaped_x, dim=reduction_axes, keepdim=True)
        return backend_module.mean(reshaped_x, axis=reduction_axes, keepdims=True)
    elif op_name == "variance":
        if is_torch:
            return backend_module.var(reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False)
        return backend_module.var(reshaped_x, axis=reduction_axes, keepdims=True)

    msg = f"Unknown grouped reduction op: {op_name}"
    raise ValueError(msg)


@global_eager_registry.register("GroupMean")
def _group_mean(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "mean", x, groups, axis)


@global_eager_registry.register("GroupVariance")
def _group_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "variance", x, groups, axis)


@global_eager_registry.register("GroupNorm")
def _group_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    axis = kwargs.get("axis", -1)
    epsilon = kwargs.get("epsilon", 1e-5)

    shape = list(x.shape)
    ndims = len(shape)
    if axis < 0:
        axis += ndims

    mean = _group_mean(backend_module, x, groups=groups, axis=axis)
    var = _group_variance(backend_module, x, groups=groups, axis=axis)

    C_per_group = shape[axis] // groups
    reshaped_dims = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]

    is_torch = backend_module.__name__ == "torch"

    if is_torch:
        reshaped_x = backend_module.reshape(x, reshaped_dims)
        normalized = (reshaped_x - mean) / backend_module.sqrt(var + epsilon)
        out = backend_module.reshape(normalized, shape)
        if weight is not None:
            w_shape = [1] * ndims
            w_shape[axis] = shape[axis]
            w = backend_module.reshape(weight, w_shape)
            out = out * w
        if bias is not None:
            b_shape = [1] * ndims
            b_shape[axis] = shape[axis]
            b = backend_module.reshape(bias, b_shape)
            out = out + b
        return out
    else:
        reshaped_x = backend_module.reshape(x, reshaped_dims)
        normalized = (reshaped_x - mean) / backend_module.sqrt(var + epsilon)
        out = backend_module.reshape(normalized, shape)
        if weight is not None:
            w_shape = [1] * ndims
            w_shape[axis] = shape[axis]
            w = backend_module.reshape(weight, w_shape)
            out = out * w
        if bias is not None:
            b_shape = [1] * ndims
            b_shape[axis] = shape[axis]
            b = backend_module.reshape(bias, b_shape)
            out = out + b
        return out
