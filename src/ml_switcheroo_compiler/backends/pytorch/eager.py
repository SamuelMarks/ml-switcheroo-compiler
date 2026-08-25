# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import cast

import torch


def _execute_accumulate_n(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_accumulate_n operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    inputs: object = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        raise ValueError("inputs must not be empty")
    res: object = inputs[0]
    for i in range(1, len(inputs)):
        res: object = res + inputs[i]
    return res


def _execute_tensor_scatter_max(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_tensor_scatter_max operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor, indices, updates = (
        args[0],
        args[1],
        args[2],
    )
    flat_idx: object = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))
    res: object = tensor.clone().flatten()
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amax", include_self=True)
    return res.reshape(tensor.shape)


def _execute_tensor_scatter_min(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_tensor_scatter_min operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor, indices, updates = (
        args[0],
        args[1],
        args[2],
    )
    flat_idx: object = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))
    res: object = tensor.clone().flatten()
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amin", include_self=True)
    return res.reshape(tensor.shape)


def _execute_tensor_scatter_update(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_tensor_scatter_update operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0].clone().index_put_(tuple(args[1].unbind(-1)), args[2])


def _execute_tensor_scatter_add(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_tensor_scatter_add operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0].clone().index_put_(tuple(args[1].unbind(-1)), args[2], accumulate=True)


def _execute_power_iteration(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_power_iteration operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    w: object = args[0]
    num_iters: object = kwargs.get("num_iters", 1)
    u: object = kwargs.get("u", None)
    if u is None:
        u: object = torch.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype, device=w.device)
    for _ in range(num_iters):
        w_t: object = w.transpose(-1, -2)
        v: object = torch.matmul(w_t, u)
        v: object = v / (torch.linalg.norm(v, dim=-2, keepdim=True) + 1e-12)
        u: object = torch.matmul(w, v)
        u: object = u / (torch.linalg.norm(u, dim=-2, keepdim=True) + 1e-12)
    sigma: object = torch.matmul(u.transpose(-1, -2), torch.matmul(w, v))
    return (v.squeeze(-1), u.squeeze(-1), sigma.squeeze(-1).squeeze(-1))


def _execute_broadcast_to(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_broadcast_to operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0].expand(kwargs["shape"])


def _execute_one_hot(*args: object, **kwargs: object) -> object:
    """_execute_one_hot function.

    Returns:
        object: Result.
    """
    inputs: object = args[0] if len(args) > 0 else kwargs.get("indices")
    depth: object = args[1] if len(args) > 1 else kwargs.get("depth")
    import torch
    import torch.nn.functional as F

    axis: object = kwargs.get("axis", -1)
    on_value: object = kwargs.get("on_value", 1.0)
    off_value: object = kwargs.get("off_value", 0.0)
    dtype_str: object = kwargs.get("dtype", "float32")
    res: object = F.one_hot(inputs.long(), num_classes=depth)
    dtype_map: object = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32, "int64": torch.int64}
    res: object = res.to(dtype_map.get(dtype_str, torch.float32))
    if on_value != 1.0 or off_value != 0.0:
        res: object = res * (on_value - off_value) + off_value
    if axis != -1 and axis != res.ndim - 1:
        res: object = res.transpose(axis, res.ndim - 1)
    return res


def _execute_cast(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cast operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensor: object = args[0]
    dtype: object = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dt_str: object = str(getattr(dtype, "value", dtype)).split(".")[-1]
    if dt_str == "float8_e4m3fn":
        dt: object = getattr(torch, "float8_e4m3fn", torch.float32)
    elif dt_str == "float8_e5m2":
        dt: object = getattr(torch, "float8_e5m2", torch.float32)
    elif "int4" in dt_str:
        dt: object = torch.int8
    elif "bfloat16" in dt_str:
        dt: object = torch.bfloat16
    elif "float16" in dt_str:
        dt: object = torch.float16
    else:
        dt: object = getattr(torch, dt_str, None)
    return tensor.to(dt)


def _execute_cummax(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cummax operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return torch.cummax(*args, **kwargs)[0]


def _execute_cummin(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cummin operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return torch.cummin(*args, **kwargs)[0]


def _execute_cumlogsumexp(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cumlogsumexp operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return torch.logcumsumexp(*args, **kwargs)


def _execute_ragged_tensor_to_dense(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_ragged_tensor_to_dense operation with proper padding and index tracking.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import torch

    rt: object = args[0]
    default_value: object = kwargs.get("default_value") if kwargs.get("default_value") is not None else (args[1] if len(args) > 1 and args[1] is not None else 0.0)

    # 1. If it's a NestedTensor
    if getattr(rt, "is_nested", False):
        # Nested tensors have a built-in to_padded_tensor
        return rt.to_padded_tensor(padding=default_value)

    # 2. If it's a list of tensors (e.g., from unbind or raw input)
    if isinstance(rt, (list, tuple)) and len(rt) > 0 and isinstance(rt[0], torch.Tensor):
        from torch.nn.utils.rnn import pad_sequence

        return pad_sequence(list(rt), batch_first=True, padding_value=float(default_value if default_value is not None else 0.0))

    # 3. If it's a dictionary or namedtuple with 'values' and 'row_splits' (standard ragged encoding)
    values: object = None
    row_splits: object = None
    if isinstance(rt, dict) and "values" in rt and "row_splits" in rt:
        values: object = rt["values"]
        row_splits: object = rt["row_splits"]
    elif hasattr(rt, "values") and hasattr(rt, "row_splits"):
        values: object = rt.values
        row_splits: object = rt.row_splits

    if values is not None and row_splits is not None:
        # Proper index tracking and shape padding
        num_rows: object = len(row_splits) - 1
        max_len: object = int(torch.max(row_splits[1:] - row_splits[:-1]).item())

        # Calculate padding
        dense_shape: object = [num_rows, max_len] + list(values.shape[1:])
        dense_tensor: object = torch.full(dense_shape, float(default_value if default_value is not None else 0.0), dtype=values.dtype, device=values.device)

        for i in range(num_rows):
            start: object = int(row_splits[i].item())
            end: object = int(row_splits[i + 1].item())
            length: object = end - start
            if length > 0:
                dense_tensor[i, :length] = values[start:end]

        return dense_tensor

    return rt


def _get_custom_torch_op_map() -> dict[str, object]:
    """Retrieve the custom torch op map property or mapping.

    Returns:
        dict: The evaluated or processed output.
    """
    return {
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


def _torch_variance(*args: object, **kwargs: object) -> object:
    """Evaluate _torch_variance operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs.setdefault("correction", kwargs.pop("ddof", 0))
    return torch.var(*args, **kwargs)


def _torch_tensordot(*args: object, **kwargs: object) -> object:
    """Evaluate _torch_tensordot operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if "axes" in kwargs:
        kwargs["dims"] = kwargs.pop("axes")
    return torch.tensordot(*args, **kwargs)


def _get_torch_reduce_op(op_type: str) -> object:
    """Get the torch.distributed reduce op.

    Args:
        op_type (str): The reduce operation type.

    Returns: object: The torch.distributed.ReduceOp.
    """
    import torch.distributed as dist

    op_type: object = op_type.lower()
    if op_type == "sum":
        return dist.ReduceOp.SUM
    elif op_type == "prod":
        return dist.ReduceOp.PRODUCT
    elif op_type == "min":
        return dist.ReduceOp.MIN
    elif op_type == "max":
        return dist.ReduceOp.MAX
    elif op_type == "avg" or op_type == "mean":
        return dist.ReduceOp.AVG
    return dist.ReduceOp.SUM


def _torch_all_gather(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Execute torch distributed AllGather.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (object): Additional args.

    Returns:
        torch.Tensor: The gathered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return torch.stack([tensor])
    tensor_list: object = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    axis: object = kwargs.get("axis", 0)
    return torch.cat(tensor_list, dim=axis)


def _torch_all_reduce(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Execute torch distributed AllReduce.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (object): Additional args.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    res: object = tensor.clone()
    op_type: object = kwargs.get("op_type", "sum")
    dist.all_reduce(res, op=_get_torch_reduce_op(op_type))
    return res


def _torch_reduce_scatter(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Execute torch distributed ReduceScatter.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (object): Additional args.

    Returns:
        torch.Tensor: The reduce-scattered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    op_type: object = kwargs.get("op_type", "sum")
    axis: object = kwargs.get("axis", 0)
    tensor_list: object = list(torch.tensor_split(tensor, dist.get_world_size(), dim=axis))
    res: object = torch.empty_like(tensor_list[0])
    dist.reduce_scatter(res, tensor_list, op=_get_torch_reduce_op(op_type))
    return res


def _torch_all_to_all(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Execute torch distributed AllToAll.

    Args:
        tensor (torch.Tensor): The input tensor.
        **kwargs (object): Additional args.

    Returns:
        torch.Tensor: The scattered/gathered tensor.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    # Assuming input is pre-split along axis 0
    split_dim: object = kwargs.get("split_axis", 0)
    concat_dim: object = kwargs.get("concat_axis", 0)
    input_tensor_list: object = list(torch.tensor_split(tensor, dist.get_world_size(), dim=split_dim))
    output_tensor_list: object = [torch.empty_like(input_tensor_list[0]) for _ in range(dist.get_world_size())]
    dist.all_to_all(output_tensor_list, input_tensor_list)
    return torch.cat(output_tensor_list, dim=concat_dim)


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Evaluate execute_op operation.

    Args:
        cls (type): The class.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        BackendNotSupportedError: An exception.
    """
    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings, resolve_target_api

    schema: object = load_backend_mappings("pytorch")
    if op_type in schema.operations and (schema.operations[op_type].target_api or schema.operations[op_type].custom_code):
        import sys

        func: object = resolve_target_api(schema.operations[op_type].target_api, schema.operations[op_type].custom_code, sys.modules[__name__])
        if func:
            return func(*args, **kwargs)
    custom_op_map: object = _get_custom_torch_op_map()
    if op_type in custom_op_map:
        return custom_op_map[op_type](*args, **kwargs)
    try:
        func: object = getattr(torch, op_type.lower())
        return func(*args, **kwargs)
    except AttributeError:
        from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

        global_func: object = global_eager_registry.get(op_type)
        if global_func is not None:
            return global_func(torch, *args, **kwargs)
        import re

        s1: object = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake: object = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        import torch.nn.functional as F

        func: object = None
        for mod in [torch, F, getattr(torch, "linalg", None), getattr(torch, "fft", None)]:
            if mod is not None and hasattr(mod, snake):
                func: object = getattr(mod, snake)
                break
        if func is not None:
            return func(*args, **kwargs)
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
