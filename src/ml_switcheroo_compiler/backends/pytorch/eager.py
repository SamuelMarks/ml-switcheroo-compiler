"""Implementation of eager operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.pytorch.eager`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast  # noqa: F401

import torch

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _execute_accumulate_n(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute accumulate n operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = getattr(inputs, "__getitem__")(0)
    for i in range(1, getattr(inputs, "__len__")()):
        res = res + getattr(inputs, "__getitem__")(i)
    return cast(EagerValue, res)


def _execute_tensor_scatter_max(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute tensor scatter max operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor, indices, updates = (
        args[0],
        args[1],
        args[2],
    )
    flat_idx = sum(cast(EagerTensorProtocol, indices)[..., i] * getattr(tensor, "stride")(i) for i in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res = getattr(getattr(tensor, "clone")(), "flatten")()
    res.scatter_reduce_(0, getattr(flat_idx, "flatten")(), getattr(updates, "flatten")(), reduce="amax", include_self=True)
    return cast(EagerValue, res.reshape(getattr(tensor, "shape")))


def _execute_tensor_scatter_min(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute tensor scatter min operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor, indices, updates = (
        args[0],
        args[1],
        args[2],
    )
    flat_idx = sum(cast(EagerTensorProtocol, indices)[..., i] * getattr(tensor, "stride")(i) for i in range(getattr(getattr(indices, "shape"), "__getitem__")(-1)))
    res = getattr(getattr(tensor, "clone")(), "flatten")()
    res.scatter_reduce_(0, getattr(flat_idx, "flatten")(), getattr(updates, "flatten")(), reduce="amin", include_self=True)
    return cast(EagerValue, res.reshape(getattr(tensor, "shape")))


def _execute_tensor_scatter_update(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute tensor scatter update operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(getattr(args[0], "clone")(), "index_put_")(tuple(getattr(args[1], "unbind")(-1)), args[2]))


def _execute_tensor_scatter_add(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute tensor scatter add operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(getattr(args[0], "clone")(), "index_put_")(tuple(getattr(args[1], "unbind")(-1)), args[2], accumulate=True))


def _execute_power_iteration(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute power iteration operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    w = args[0]
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = torch.ones(getattr(w, "shape")[:-2] + (getattr(w, "shape")[-2], 1), dtype=getattr(w, "dtype"), device=getattr(w, "device"))
    for _ in range(cast("int", num_iters)):
        w_t = getattr(w, "transpose")(-1, -2)
        v = getattr(torch, "matmul")(w_t, u)
        v = v / (torch.linalg.norm(v, dim=-2, keepdim=True) + 1e-12)
        u = getattr(torch, "matmul")(w, v)
        u = u / (torch.linalg.norm(u, dim=-2, keepdim=True) + 1e-12)
    sigma = getattr(torch, "matmul")(getattr(u, "transpose")(-1, -2), getattr(torch, "matmul")(w, v))
    return (getattr(v, "squeeze")(-1), getattr(u, "squeeze")(-1), getattr(getattr(sigma, "squeeze")(-1), "squeeze")(-1))


def _execute_broadcast_to(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute broadcast to operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(args[0], "expand")(kwargs["shape"]))


def _execute_one_hot(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute one hot operation on the active backend array.

    Args:
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("indices")
    depth = args[1] if len(args) > 1 else kwargs.get("depth")
    import torch
    import torch.nn.functional as F

    axis = kwargs.get("axis", -1)
    on_value = kwargs.get("on_value", 1.0)
    off_value = kwargs.get("off_value", 0.0)
    dtype_str = kwargs.get("dtype", "float32")
    res = getattr(F, "one_hot")(getattr(inputs, "long")(), num_classes=depth)
    dtype_map = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32, "int64": torch.int64}
    res = getattr(res, "to")(getattr(dtype_map, "get")(dtype_str, torch.float32))
    if on_value != 1.0 or off_value != 0.0:
        res = res * (cast(EagerTensorProtocol, on_value) - cast(EagerTensorProtocol, off_value)) + cast(EagerTensorProtocol, off_value)
    if axis != -1 and axis != res.ndim - 1:
        res = res.transpose(axis, res.ndim - 1)
    return cast(EagerValue, res)


def _execute_cast(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute cast operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor = args[0]
    dtype = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dt_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
    if dt_str == "float8_e4m3fn":
        dt = getattr(torch, "float8_e4m3fn", torch.float32)
    elif dt_str == "float8_e5m2":
        dt = getattr(torch, "float8_e5m2", torch.float32)
    elif "int4" in dt_str:
        dt = torch.int8
    elif "bfloat16" in dt_str:
        dt = torch.bfloat16
    elif "float16" in dt_str:
        dt = torch.float16
    else:
        dt = getattr(torch, dt_str, None)
    return cast(EagerValue, getattr(tensor, "to")(dt))


def _execute_cummax(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute cummax operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(torch, "cummax")(*args, **kwargs)[0])


def _execute_cummin(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute cummin operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(torch, "cummin")(*args, **kwargs)[0])


def _execute_cumlogsumexp(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager execute cumlogsumexp operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(torch, "logcumsumexp")(*args, **kwargs))


def _execute_ragged_tensor_to_dense(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _execute_ragged_tensor_to_dense operation with proper padding and index tracking.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import torch

    rt = args[0]
    default_value = kwargs.get("default_value") if kwargs.get("default_value") is not None else (args[1] if len(args) > 1 and args[1] is not None else 0.0)

    # 1. If it's a NestedTensor
    if getattr(rt, "is_nested", False):
        # Nested tensors have a built-in to_padded_tensor
        return cast(EagerValue, getattr(rt, "to_padded_tensor")(padding=default_value))

    # 2. If it's a list of tensors (e.g., from unbind or raw input)
    if isinstance(rt, (list, tuple)) and len(rt) > 0 and isinstance(rt[0], torch.Tensor):
        from torch.nn.utils.rnn import pad_sequence

        return pad_sequence(list(rt), batch_first=True, padding_value=float(cast(float, default_value if default_value is not None else 0.0)))

    # 3. If it's a dictionary or namedtuple with 'values' and 'row_splits' (standard ragged encoding)
    values = None
    row_splits = None
    if isinstance(rt, dict) and "values" in rt and "row_splits" in rt:
        values = rt["values"]
        row_splits = rt["row_splits"]
    elif hasattr(rt, "values") and hasattr(rt, "row_splits"):
        values = getattr(rt, "values")
        row_splits = getattr(rt, "row_splits")

    if values is not None and row_splits is not None:
        # Proper index tracking and shape padding
        num_rows = len(row_splits) - 1
        max_len = int(torch.max(row_splits[1:] - row_splits[:-1]).item())

        # Calculate padding
        dense_shape = [num_rows, max_len] + list(values.shape[1:])
        dense_tensor = torch.full(dense_shape, float(cast(float, default_value if default_value is not None else 0.0)), dtype=values.dtype, device=values.device)

        for i in range(num_rows):
            start = int(row_splits[i].item())
            end = int(row_splits[i + 1].item())
            length = end - start
            if length > 0:
                dense_tensor[i, :length] = values[start:end]

        return dense_tensor

    return rt


def _get_custom_torch_op_map() -> dict[str, Callable[..., EagerValue]]:
    """Retrieve the custom torch op map property or mapping.

    Returns:
        dict: The evaluated or processed output.
    """
    return {
        "AllGather": _torch_all_gather,
        "AllReduce": _torch_all_reduce,
        "NcclAllReduce": _torch_all_reduce,
        "HierarchicalCopyAllReduce": _torch_all_reduce,
        "ReduceScatter": _torch_reduce_scatter,
        "AllToAll": _torch_all_to_all,
        "RaggedTensorToDense": _execute_ragged_tensor_to_dense,
        "TensorScatterUpdate": _execute_tensor_scatter_update,
        "TensorScatterAdd": _execute_tensor_scatter_add,
        "TensorScatterMax": _execute_tensor_scatter_max,
        "TensorScatterMin": _execute_tensor_scatter_min,
        "PowerIteration": _execute_power_iteration,
        "BroadcastTo": _execute_broadcast_to,
        "Cummax": _execute_cummax,
        "Cummin": _execute_cummin,
        "Cumlogsumexp": _execute_cumlogsumexp,
        "Cast": _execute_cast,
        "OneHot": _execute_one_hot,
    }


def _torch_variance(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager torch variance operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs.setdefault("correction", kwargs.pop("ddof", 0))
    return cast(EagerValue, getattr(torch, "var")(*args, **kwargs))


def _torch_tensordot(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager torch tensordot operation on the active backend array.

    Args:
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if "axes" in kwargs:
        kwargs["dims"] = kwargs.pop("axes")
    return cast(EagerValue, getattr(torch, "tensordot")(*args, **kwargs))


def _get_torch_reduce_op(op_type: str) -> EagerValue:
    """Get the torch.distributed reduce op.

    Args:
        op_type (str): The reduce operation type.

    Returns: torch.Tensor: The torch.distributed.ReduceOp.
    """
    import torch.distributed as dist

    op_type = op_type.lower()
    if op_type == "sum":
        return cast(EagerValue, dist.ReduceOp.SUM)
    elif op_type == "prod":
        return cast(EagerValue, dist.ReduceOp.PRODUCT)
    elif op_type == "min":
        return cast(EagerValue, dist.ReduceOp.MIN)
    elif op_type == "max":
        return cast(EagerValue, dist.ReduceOp.MAX)
    elif op_type == "avg" or op_type == "mean":
        return cast(EagerValue, dist.ReduceOp.AVG)
    return cast(EagerValue, dist.ReduceOp.SUM)


def _torch_all_gather(tensor: torch.Tensor, **kwargs: EagerValue) -> torch.Tensor:
    """Execute torch distributed AllGather.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (EagerValue): Additional args.

    Returns:
        torch.Tensor: The gathered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return torch.stack([tensor])
    tensor_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    axis = kwargs.get("axis", 0)
    return torch.cat(tensor_list, dim=axis)


def _torch_all_reduce(tensor: torch.Tensor, **kwargs: EagerValue) -> torch.Tensor:
    """Execute torch distributed AllReduce.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (EagerValue): Additional args.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    res = tensor.clone()
    op_type = kwargs.get("op_type", "sum")
    dist.all_reduce(res, op=_get_torch_reduce_op(cast("str", op_type)))
    return res


def _torch_reduce_scatter(tensor: torch.Tensor, **kwargs: EagerValue) -> torch.Tensor:
    """Execute torch distributed ReduceScatter.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (EagerValue): Additional args.

    Returns:
        torch.Tensor: The reduce-scattered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    op_type = kwargs.get("op_type", "sum")
    axis = kwargs.get("axis", 0)
    tensor_list = list(getattr(torch, "tensor_split")(tensor, dist.get_world_size(), dim=axis))
    res = torch.empty_like(tensor_list[0])
    dist.reduce_scatter(res, tensor_list, op=_get_torch_reduce_op(cast("str", op_type)))
    return res


def _torch_all_to_all(tensor: torch.Tensor, **kwargs: EagerValue) -> torch.Tensor:
    """Execute torch distributed AllToAll.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (EagerValue): Additional args.

    Returns:
        torch.Tensor: The scattered/gathered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    # Assuming input is pre-split along axis 0
    split_dim = kwargs.get("split_axis", 0)
    concat_dim = kwargs.get("concat_axis", 0)
    input_tensor_list = list(getattr(torch, "tensor_split")(tensor, dist.get_world_size(), dim=split_dim))
    output_tensor_list = [torch.empty_like(input_tensor_list[0]) for _ in range(dist.get_world_size())]
    dist.all_to_all(output_tensor_list, input_tensor_list)
    return torch.cat(output_tensor_list, dim=concat_dim)


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
    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings, resolve_target_api

    schema = load_backend_mappings("pytorch")
    if op_type in schema.operations and (schema.operations[op_type].target_api or schema.operations[op_type].custom_code):
        import sys

        func = resolve_target_api(schema.operations[op_type].target_api, schema.operations[op_type].custom_code, sys.modules[__name__])
        if func:
            return cast(Callable[..., EagerValue], func)(*args, **kwargs)
    custom_op_map = _get_custom_torch_op_map()
    if op_type in custom_op_map:
        return custom_op_map[op_type](*args, **kwargs)
    try:
        func = getattr(torch, op_type.lower())
        return cast(EagerValue, func(*args, **kwargs))
    except AttributeError:
        from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

        global_func = global_eager_registry.get(op_type)
        if global_func is not None:
            return global_func(torch, *args, **kwargs)
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        import torch.nn.functional as F

        func = None
        for mod in [torch, F, getattr(torch, "linalg", None), getattr(torch, "fft", None)]:
            if mod is not None and hasattr(mod, snake):
                func = getattr(mod, snake)
                break
        if func is not None:
            return cast(EagerValue, func(*args, **kwargs))
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
