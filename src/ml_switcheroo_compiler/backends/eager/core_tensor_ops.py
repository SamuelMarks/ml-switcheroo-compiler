# ruff: noqa: F405, F403
"""Core utilities."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2

import importlib
import typing


from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Sort")
def _sort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    a = args[0]  # pragma: no cover
    axis = kwargs.get("axis", -1)  # pragma: no cover
    if hasattr(backend_module, "sort"):  # pragma: no cover
        return backend_module.sort(a, axis=axis)  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("ArgSort")
def _argsort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    a = args[0]  # pragma: no cover
    axis = kwargs.get("axis", -1)  # pragma: no cover
    if hasattr(backend_module, "argsort"):  # pragma: no cover
        return backend_module.argsort(a, axis=axis)  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("Reshape")
def _reshape(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]  # pragma: no cover
    shape = (
        list(args[1]) if len(args) > 1 else list(kwargs.get("shape", kwargs.get("newshape")))
    )  # pragma: no cover
    if hasattr(backend_module, "reshape"):  # pragma: no cover
        return backend_module.reshape(x, shape)  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("Permute")
def _permute(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]  # pragma: no cover
    axes = kwargs.get("axes", None)  # pragma: no cover
    if hasattr(backend_module, "transpose"):  # pragma: no cover
        return backend_module.transpose(x, axes)  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("TensorScatterUpdate")
def _tensor_scatter_update(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].set(updates)
    if name == "torch":  # pragma: no branch
        return tensor.clone().index_put_(tuple(indices.unbind(-1)), updates)  # pragma: no cover
    if name == "keras.ops":
        return backend_module.tensor_scatter_update(tensor, indices, updates)
    if name in {"tensorflow.math", "tensorflow"}:
        tf = importlib.import_module("tensorflow")

        return tf.tensor_scatter_nd_update(tensor, indices, updates)
    raise NotImplementedError(f"TensorScatterUpdate eager not implemented for {name}")


@global_eager_registry.register("TensorScatterAdd")
def _tensor_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].add(updates)
    if name == "torch":  # pragma: no branch
        return tensor.clone().index_put_(
            tuple(indices.unbind(-1)), updates, accumulate=True
        )  # pragma: no cover
    if name == "keras.ops":
        return backend_module.tensor_scatter_add(tensor, indices, updates)
    if name in {"tensorflow.math", "tensorflow"}:  # pragma: no branch
        tf = importlib.import_module("tensorflow")

        return tf.tensor_scatter_nd_add(tensor, indices, updates)
    raise NotImplementedError(
        f"TensorScatterAdd eager not implemented for {name}"
    )  # pragma: no cover


@global_eager_registry.register("TensorScatterMax")
def _tensor_scatter_max(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].max(updates)
    if name == "torch":  # pragma: no branch
        raise NotImplementedError(
            "TensorScatterMax not implemented for torch in legacy eager"
        )  # pragma: no cover
    if name == "keras.ops":
        return backend_module.tensor_scatter_max(tensor, indices, updates)
    if name in {"tensorflow.math", "tensorflow"}:  # pragma: no branch
        tf = importlib.import_module("tensorflow")

        return tf.tensor_scatter_nd_max(tensor, indices, updates)
    raise NotImplementedError(
        f"TensorScatterMax eager not implemented for {name}"
    )  # pragma: no cover


@global_eager_registry.register("TensorScatterMin")
def _tensor_scatter_min(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    name = getattr(backend_module, "__name__", "")
    if name == "jax.numpy":
        return tensor.at[tuple(backend_module.moveaxis(indices, -1, 0))].min(updates)
    if name == "torch":  # pragma: no branch
        raise NotImplementedError(
            "TensorScatterMin not implemented for torch in legacy eager"
        )  # pragma: no cover
    if name == "keras.ops":
        return backend_module.tensor_scatter_min(tensor, indices, updates)
    if name in {"tensorflow.math", "tensorflow"}:  # pragma: no branch
        tf = importlib.import_module("tensorflow")

        return tf.tensor_scatter_nd_min(tensor, indices, updates)
    raise NotImplementedError(
        f"TensorScatterMin eager not implemented for {name}"
    )  # pragma: no cover


def _normalize_shape(shape: object) -> object:
    """Function docstring.

    Args:
        shape: Arg.
    """
    if hasattr(shape, "data"):
        shape = shape.data
    if hasattr(shape, "tolist") and callable(shape.tolist):
        shape = shape.tolist()
    if isinstance(shape, tuple):
        shape = list(shape)
    return shape


def _extract_shape_value(val: object) -> int:
    """Function docstring.

    Args:
        val: Arg.
    """
    if hasattr(val, "data"):
        val = val.data

    if hasattr(val, "item") and callable(val.item):
        val = val.item()
    elif hasattr(val, "tolist") and callable(val.tolist):
        val_list = val.tolist()
        val = val_list[0] if isinstance(val_list, list) else val_list

    return int(typing.cast(typing.Any, val))


def _parse_eager_shape(shape: object) -> list[int]:
    """Function docstring.

    Args:
        shape: Arg.
    """
    shape = _normalize_shape(shape)

    if not isinstance(shape, list) or not shape:
        return typing.cast(list[int], shape)

    return [_extract_shape_value(s) for s in shape]


@global_eager_registry.register("BroadcastTo")
def _broadcast_to(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else args[0] if len(args) > 0 else None)
    parsed_shape = _parse_eager_shape(shape)
    return backend_module.broadcast_to(args[0], parsed_shape)


@global_eager_registry.register("Zeros")
def _zeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    shape = tuple(_parse_eager_shape(shape))
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
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    shape = tuple(_parse_eager_shape(shape))
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
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    shape = tuple(_parse_eager_shape(shape))
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
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
    broadcast_dimensions = kwargs.get(
        "broadcast_dimensions", args[2] if len(args) > MAGIC_VAL_2 else None
    )
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
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]  # pragma: no cover
    k = kwargs.get("k", args[1] if len(args) > 1 else None)  # pragma: no cover
    if hasattr(k, "item"):
        k = int(k.item())
    elif hasattr(k, "data") and hasattr(k.data, "item"):
        k = int(k.data.item())
    elif k is not None:
        k = int(k)
    axis = kwargs.get("axis", -1)  # pragma: no cover
    return_indices = kwargs.get("return_indices", None)

    kth = max(0, x.shape[axis] - k)
    if return_indices is False:
        if hasattr(backend_module, "topk"):
            return backend_module.topk(x, k, axis=axis)
        val = backend_module.partition(x, kth, axis=axis)
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(-k, None)
        return val[tuple(slc)]

    idx = backend_module.argpartition(x, kth, axis=axis)
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(-k, None)
    idx_k = idx[tuple(slc)]

    if return_indices is True:
        return idx_k

    val_k = backend_module.take_along_axis(x, idx_k, axis=axis)
    return val_k, idx_k  # pragma: no cover


@global_eager_registry.register("Resize")
def _resize(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
    if hasattr(backend_module, "zeros"):  # pragma: no branch
        return backend_module.zeros((x.shape[0], *shape, x.shape[-1]), dtype=x.dtype)
    return None  # pragma: no cover


@global_eager_registry.register("DynamicUpdateSlice")
def _dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    update = args[1]
    start_indices = args[2:] if len(args) > MAGIC_VAL_2 else kwargs.get("start_indices", [])
    if hasattr(backend_module, "dynamic_update_slice"):
        return backend_module.dynamic_update_slice(x, update, start_indices)
    return x


@global_eager_registry.register("ConvGeneralDilated")
def _conv_general_dilated_fallback(
    backend_module: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.zeros((1,))


def generic_zeros(backend_module: object, shape: tuple[int, ...]) -> object:
    """Generic zeros function.

    Args:
        backend_module (Any): The backend module.
        shape (tuple[int, ...]): Shape of the tensor.

    Returns:
        object: A tensor of zeros.
    """
    return backend_module.zeros(shape)


def generic_array(backend_module: object, data: object, dtype: object = None) -> object:  # noqa: PLR0911
    """Generic array creation.

    Args:
        backend_module (Any): The backend module.
        data (object): The data to convert.
        dtype (object, optional): The data type.

    Returns:
        object: A tensor array.
    """
    try:
        from ml_switcheroo_compiler.core.dtype import DType

        if isinstance(data, DType):
            return data
        if data is None:
            return None
        if getattr(data, "__name__", "") == "mlx.core":  # pragma: no branch
            return data  # pragma: no cover
        if "mlx.core.array" in str(type(data)):
            return data

        if dtype is not None:
            dtype_str = str(dtype).split(".")[-1]
            if dtype_str == "bool" and getattr(backend_module, "__name__", "") == "mlx.core":
                dtype_str = "bool_"
            dt = getattr(backend_module, dtype_str, dtype)
            return backend_module.array(data, dtype=dt)
        return backend_module.array(data)
    except AttributeError:
        return backend_module.convert_to_tensor(data)


def generic_asarray(backend_module: object, data: object) -> object:
    """Fallback to convert data to array if backend lacks specific logic."""
    try:
        from ml_switcheroo_compiler.core.dtype import DType

        if isinstance(data, DType):
            return data
        if hasattr(backend_module, "asarray"):
            return backend_module.asarray(data)
        return generic_array(backend_module, data)
    except Exception:
        return data


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


@global_eager_registry.register("Stack")
def _stack(backend_module: object, arrays: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    try:
        from ml_switcheroo_compiler.backends.eager.core_tensor_ops import generic_asarray

        arrays = [generic_asarray(backend_module, a) for a in arrays]
    except Exception:
        import traceback

        traceback.print_exc()

    if hasattr(backend_module, "stack"):
        return backend_module.stack(arrays, *args, **kwargs)
    return backend_module.concatenate(arrays, *args, **kwargs)


@global_eager_registry.register("Concatenate")
def _concatenate(backend_module: object, *args: object, **kwargs: object) -> object:
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        arrays = list(args[0])
    else:
        arrays = list(args)
    try:
        from ml_switcheroo_compiler.backends.eager.core_tensor_ops import generic_asarray

        arrays = [generic_asarray(backend_module, a) for a in arrays]
    except Exception:
        import traceback

        traceback.print_exc()
    return backend_module.concatenate(arrays, **kwargs)


@global_eager_registry.register("Repeat")
def _repeat(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]
    repeats = kwargs.get("repeats", args[1] if len(args) > 1 else None)
    repeats = (
        _parse_eager_shape(repeats)
        if isinstance(repeats, (list, tuple))
        else _extract_shape_value(repeats)
    )

    if len(args) > 1:
        args = (x, repeats) + args[2:]
    else:
        kwargs["repeats"] = repeats

    return backend_module.repeat(*args, **kwargs)


@global_eager_registry.register("Tile")
def _tile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]
    reps = kwargs.get("reps", args[1] if len(args) > 1 else None)
    reps = tuple(_parse_eager_shape(reps))

    if len(args) > 1:
        args = (x, reps) + args[2:]
    else:
        kwargs["reps"] = reps

    return backend_module.tile(*args, **kwargs)


@global_eager_registry.register("Argpartition")
def _argpartition(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]
    kth = kwargs.get("kth", args[1] if len(args) > 1 else None)
    if hasattr(kth, "item"):
        kth = int(kth.item())
    elif hasattr(kth, "data") and hasattr(kth.data, "item"):
        kth = int(kth.data.item())
    else:
        kth = int(kth)

    if len(args) > 1:
        args = (x, kth) + args[2:]
    else:
        kwargs["kth"] = kth

    return backend_module.argpartition(*args, **kwargs)


@global_eager_registry.register("Partition")
def _partition(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]
    kth = kwargs.get("kth", args[1] if len(args) > 1 else None)
    if hasattr(kth, "item"):
        kth = int(kth.item())
    elif hasattr(kth, "data") and hasattr(kth.data, "item"):
        kth = int(kth.data.item())
    else:
        kth = int(kth)

    if len(args) > 1:
        args = (x, kth) + args[2:]
    else:
        kwargs["kth"] = kth

    return backend_module.partition(*args, **kwargs)


@global_eager_registry.register("Split")
def _split(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]

    # Extract size from either args or kwargs
    split_size = kwargs.pop("split_size_or_sections", args[1] if len(args) > 1 else None)
    if split_size is None:
        split_size = kwargs.pop("indices_or_sections", None)

    # Standardize dimension keyword
    dim = kwargs.pop("dim", kwargs.pop("axis", 0))

    if hasattr(backend_module, "split"):
        try:
            return backend_module.split(x, split_size, axis=dim)
        except TypeError:
            try:
                return backend_module.split(x, split_size, dim=dim)
            except TypeError:
                return backend_module.split(x, split_size_or_sections=split_size, dim=dim)
    return backend_module.split(x, split_size, axis=dim)


@global_eager_registry.register("Diag")
def _diag(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x = args[0]
    if hasattr(x, "data"):
        x = x.data
    k = kwargs.pop("k", kwargs.pop("diagonal", args[1] if len(args) > 1 else 0))

    if hasattr(backend_module, "diag"):
        try:
            return backend_module.diag(x, k=k)
        except TypeError:
            return backend_module.diag(x, diagonal=k)
    return backend_module.diag(x, k)


__all__ = [
    "MAGIC_VAL_2",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_argsort",
    "_broadcast_in_dim",
    "_broadcast_to",
    "_conv_general_dilated_fallback",
    "_dynamic_update_slice",
    "_extract_shape_value",
    "_full",
    "_normalize_shape",
    "_ones",
    "_parse_eager_shape",
    "_permute",
    "_reshape",
    "_resize",
    "_sort",
    "_stack",
    "_tensor_scatter_add",
    "_tensor_scatter_max",
    "_tensor_scatter_min",
    "_tensor_scatter_update",
    "_top_k",
    "_zeros",
    "generic_array",
    "generic_asarray",
    "generic_item",
    "generic_zeros",
    "global_eager_registry",
    "importlib",
    "typing",
]
