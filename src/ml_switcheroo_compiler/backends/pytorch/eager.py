"""Operations and utilities for `ml_switcheroo_compiler.backends.pytorch.eager`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_array,
    coerce_float,
    coerce_int,
    coerce_numeric,
    coerce_operand_list,
    coerce_shape,
    coerce_str,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

_has_torch = sys.modules.get("torch") is not None or (
    "torch" not in sys.modules and importlib.util.find_spec("torch") is not None
)
torch: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("torch") if _has_torch else None
F: types.ModuleType | BackendModuleProtocol | None = (
    importlib.import_module("torch.nn.functional") if _has_torch else None
)


def _as_pt_tensor(val: BackendArg) -> EagerTensorProtocol:
    """Convert or narrow `val` to a PyTorch-compatible `EagerTensorProtocol`.

    Args:
        val (BackendArg): Input tensor or array-like operand.

    Returns:
        EagerTensorProtocol: Adapted PyTorch tensor protocol.
    """
    return adapt_backend_module(torch).require_op("as_tensor")(val)


def _execute_accumulate_n(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute accumulate n operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    raw_inputs = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    seq = coerce_operand_list(raw_inputs)
    if not seq:
        raise ValueError("inputs must not be empty")
    res: BackendArg = seq[0]
    for i in range(1, len(seq)):
        res = coerce_numeric(res) + coerce_numeric(seq[i])
    return res


def _execute_tensor_scatter_max(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute tensor scatter max operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = _as_pt_tensor(args[0])
    indices = _as_pt_tensor(args[1])
    updates = _as_pt_tensor(args[2])
    ndim_last = int(coerce_int(coerce_shape(indices.shape)[-1]))
    flat_idx: EagerTensorProtocol = _as_pt_tensor(0)
    for i in range(ndim_last):
        stride_i = coerce_int(adapt_backend_module(tensor).require_op("stride")(i))
        term = indices[..., i] * stride_i
        flat_idx = term if i == 0 else (flat_idx + term)
    cloned: EagerTensorProtocol = adapt_backend_module(tensor).require_op("clone")()
    res: EagerTensorProtocol = adapt_backend_module(cloned).require_op("flatten")()
    adapt_backend_module(res).require_op("scatter_reduce_")(
        0,
        adapt_backend_module(flat_idx).require_op("flatten")(),
        adapt_backend_module(updates).require_op("flatten")(),
        reduce="amax",
        include_self=True,
    )
    return adapt_backend_module(res).require_op("reshape")(tensor.shape)


def _execute_tensor_scatter_min(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute tensor scatter min operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = _as_pt_tensor(args[0])
    indices = _as_pt_tensor(args[1])
    updates = _as_pt_tensor(args[2])
    ndim_last = int(coerce_int(coerce_shape(indices.shape)[-1]))
    flat_idx: EagerTensorProtocol = _as_pt_tensor(0)
    for i in range(ndim_last):
        stride_i = coerce_int(adapt_backend_module(tensor).require_op("stride")(i))
        term = indices[..., i] * stride_i
        flat_idx = term if i == 0 else (flat_idx + term)
    cloned: EagerTensorProtocol = adapt_backend_module(tensor).require_op("clone")()
    res: EagerTensorProtocol = adapt_backend_module(cloned).require_op("flatten")()
    adapt_backend_module(res).require_op("scatter_reduce_")(
        0,
        adapt_backend_module(flat_idx).require_op("flatten")(),
        adapt_backend_module(updates).require_op("flatten")(),
        reduce="amin",
        include_self=True,
    )
    return adapt_backend_module(res).require_op("reshape")(tensor.shape)


def _execute_tensor_scatter_update(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute tensor scatter update operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    cloned = adapt_backend_module(args[0]).require_op("clone")()
    unbound = tuple(coerce_operand_list(adapt_backend_module(args[1]).require_op("unbind")(-1)))
    return adapt_backend_module(cloned).require_op("index_put_")(unbound, args[2])


def _execute_tensor_scatter_add(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute tensor scatter add operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    cloned = adapt_backend_module(args[0]).require_op("clone")()
    unbound = tuple(coerce_operand_list(adapt_backend_module(args[1]).require_op("unbind")(-1)))
    return adapt_backend_module(cloned).require_op("index_put_")(unbound, args[2], accumulate=True)


def _execute_power_iteration(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute power iteration operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    torch_adp = adapt_backend_module(torch)
    linalg_adp = adapt_backend_module(torch_adp.require_symbol("linalg"))
    w_raw = args[0]
    w: BackendArg = _as_pt_tensor(w_raw) if isinstance(w_raw, (int, float, bool, list, tuple)) else w_raw
    w_adp = adapt_backend_module(w)
    num_iters = kwargs.get("num_iters", 1)
    u_raw = kwargs.get("u", None)
    w_shape = coerce_shape(w_adp.require_symbol("shape"))
    u: BackendArg = (
        (_as_pt_tensor(u_raw) if isinstance(u_raw, (int, float, bool, list, tuple)) else u_raw)
        if u_raw is not None
        else torch_adp.require_op("ones")(
            w_shape[:-2] + (w_shape[-2], 1),
            dtype=w_adp.get_symbol("dtype", None),
            device=w_adp.get_symbol("device", None),
        )
    )
    v: BackendArg = u
    matmul_fn = torch_adp.require_op("matmul")
    norm_fn = linalg_adp.require_op("norm")
    for _ in range(coerce_int(num_iters)):
        w_t = w_adp.require_op("transpose")(-1, -2)
        v = matmul_fn(w_t, u)
        v = _as_pt_tensor(v) / (_as_pt_tensor(norm_fn(v, dim=-2, keepdim=True)) + 1e-12)
        u = matmul_fn(w, v)
        u = _as_pt_tensor(u) / (_as_pt_tensor(norm_fn(u, dim=-2, keepdim=True)) + 1e-12)
    sigma = matmul_fn(adapt_backend_module(u).require_op("transpose")(-1, -2), matmul_fn(w, v))
    sig_sq1 = adapt_backend_module(sigma).require_op("squeeze")(-1)
    return (
        adapt_backend_module(v).require_op("squeeze")(-1),
        adapt_backend_module(u).require_op("squeeze")(-1),
        adapt_backend_module(sig_sq1).require_op("squeeze")(-1),
    )


def _execute_broadcast_to(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute broadcast to operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(args[0]).require_op("expand")(kwargs["shape"])


def _execute_one_hot(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute one hot operation on the active backend array.

    Args:
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("indices")
    depth = args[1] if len(args) > 1 else kwargs.get("depth")
    import torch
    import torch.nn.functional as F

    axis = coerce_int(kwargs.get("axis", -1))
    on_value = coerce_numeric(kwargs.get("on_value", 1.0))
    off_value = coerce_numeric(kwargs.get("off_value", 0.0))
    dtype_str = coerce_str(kwargs.get("dtype", "float32"))
    long_inp = adapt_backend_module(inputs).require_op("long")()
    res: EagerTensorProtocol = require_callable(F.one_hot)(long_inp, num_classes=depth)
    dtype_map = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32, "int64": torch.int64}
    res = adapt_backend_module(res).require_op("to")(dtype_map.get(dtype_str, torch.float32))
    if on_value != 1.0 or off_value != 0.0:
        res = res * (on_value - off_value) + off_value
    if axis != -1 and axis != res.ndim - 1:
        res = adapt_backend_module(res).require_op("transpose")(axis, res.ndim - 1)
    return res


def _execute_cast(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute cast operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = args[0]
    dtype = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dt_str = str(adapt_backend_module(dtype).get_symbol("value", dtype)).split(".")[-1]
    torch_adp = adapt_backend_module(torch)
    f32 = torch_adp.get_symbol("float32", None)
    dt: BackendArg
    if dt_str == "float8_e4m3fn":
        dt = torch_adp.get_symbol("float8_e4m3fn", f32)
    elif dt_str == "float8_e5m2":
        dt = torch_adp.get_symbol("float8_e5m2", f32)
    elif "int4" in dt_str:
        dt = torch_adp.get_symbol("int8", None)
    elif "bfloat16" in dt_str:
        dt = torch_adp.get_symbol("bfloat16", None)
    elif "float16" in dt_str:
        dt = torch_adp.get_symbol("float16", None)
    else:
        dt = torch_adp.get_symbol(dt_str, None)
    return adapt_backend_module(tensor).require_op("to")(dt)


def _execute_cummax(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute cummax operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(torch).require_op("cummax")(*args, **kwargs)[0]


def _execute_cummin(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute cummin operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(torch).require_op("cummin")(*args, **kwargs)[0]


def _execute_cumlogsumexp(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute cumlogsumexp operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(torch).require_op("logcumsumexp")(*args, **kwargs)


def _execute_ragged_tensor_to_dense(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Evaluate _execute_ragged_tensor_to_dense operation with proper padding and index tracking.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import torch

    rt = args[0]
    default_value = (
        kwargs.get("default_value")
        if kwargs.get("default_value") is not None
        else (args[1] if len(args) > 1 and args[1] is not None else 0.0)
    )

    # 1. If it's a NestedTensor
    if adapt_backend_module(rt).get_symbol("is_nested", False):
        # Nested tensors have a built-in to_padded_tensor
        return adapt_backend_module(rt).require_op("to_padded_tensor")(padding=default_value)

    # 2. If it's a list of tensors (e.g., from unbind or raw input)
    if isinstance(rt, (list, tuple)) and len(rt) > 0 and isinstance(rt[0], torch.Tensor):
        from torch.nn.utils.rnn import pad_sequence

        return require_callable(pad_sequence)(
            list(rt),
            batch_first=True,
            padding_value=float(coerce_float(default_value if default_value is not None else 0.0)),
        )

    # 3. If it's a dictionary or namedtuple with 'values' and 'row_splits' (standard ragged encoding)
    values = None
    row_splits = None
    if isinstance(rt, dict) and "values" in rt and "row_splits" in rt:
        values = rt["values"]
        row_splits = rt["row_splits"]
    elif adapt_backend_module(rt).has_symbol("values") and adapt_backend_module(rt).has_symbol("row_splits"):
        values = adapt_backend_module(rt).require_symbol("values")
        row_splits = adapt_backend_module(rt).require_symbol("row_splits")

    if values is not None and row_splits is not None:
        val_t = coerce_array(torch, values)
        splits_t = coerce_array(torch, row_splits)
        num_rows = int(splits_t.shape[0]) - 1
        diffs = splits_t[1:] - splits_t[:-1]
        max_item = adapt_backend_module(require_callable(torch.max)(diffs)).require_op("item")()
        max_len = int(coerce_int(max_item))

        # Calculate padding
        dense_shape = [num_rows, max_len] + [int(d) for d in val_t.shape[1:]]
        dense_tensor: EagerTensorProtocol = require_callable(torch.full)(
            dense_shape,
            float(coerce_float(default_value if default_value is not None else 0.0)),
            dtype=val_t.dtype,
            device=adapt_backend_module(val_t).get_symbol("device", None),
        )

        for i in range(num_rows):
            start = int(coerce_int(adapt_backend_module(splits_t[i]).require_op("item")()))
            end = int(coerce_int(adapt_backend_module(splits_t[i + 1]).require_op("item")()))
            length = end - start
            if length > 0:
                dense_tensor[i, :length] = val_t[start:end]

        return dense_tensor

    return rt


def _get_custom_torch_op_map() -> dict[str, TypedVariadicCallable[BackendArg]]:
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


def _torch_variance(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager torch variance operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    kwargs.setdefault("correction", kwargs.pop("ddof", 0))
    return adapt_backend_module(torch).require_op("var")(*args, **kwargs)


def _torch_tensordot(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager torch tensordot operation on the active backend array.

    Args:
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if "axes" in kwargs:
        kwargs["dims"] = kwargs.pop("axes")
    return adapt_backend_module(torch).require_op("tensordot")(*args, **kwargs)


def _get_torch_reduce_op(op_type: str) -> BackendArg:
    """Get the torch.distributed reduce op.

    Args:
        op_type (str): The reduce operation type.

    Returns: torch.Tensor: The torch.distributed.ReduceOp.
    """
    import torch.distributed as dist

    op_type = op_type.lower()
    raw_reduce_op: BackendArg = dist.ReduceOp
    red_adp = adapt_backend_module(raw_reduce_op)
    if op_type == "sum":
        return red_adp.require_symbol("SUM")
    elif op_type == "prod":
        return red_adp.require_symbol("PRODUCT")
    elif op_type == "min":
        return red_adp.require_symbol("MIN")
    elif op_type == "max":
        return red_adp.require_symbol("MAX")
    elif op_type == "avg" or op_type == "mean":
        return red_adp.require_symbol("AVG")
    return red_adp.require_symbol("SUM")


def _torch_all_gather(*args: BackendArg, **kwargs: BackendArg) -> EagerTensorProtocol:
    """Execute torch distributed AllGather.

    Args:
        *args (BackendArg): Positional args, first is input tensor.
        **kwargs (BackendArg): Additional args.

    Returns:
        EagerTensorProtocol: The gathered tensor.
    """
    import torch.distributed as dist

    torch_adp = adapt_backend_module(torch)
    tensor = _as_pt_tensor(args[0])
    if not dist.is_initialized():
        return torch_adp.require_op("stack")([tensor])
    empty_like_fn = torch_adp.require_op("empty_like")
    tensor_list = [empty_like_fn(tensor) for _ in range(int(coerce_int(dist.get_world_size())))]
    dist.all_gather(tensor_list, tensor)
    axis = int(coerce_int(kwargs.get("axis", 0)))
    return torch_adp.require_op("cat")(tensor_list, dim=axis)


def _torch_all_reduce(*args: BackendArg, **kwargs: BackendArg) -> EagerTensorProtocol:
    """Execute torch distributed AllReduce.

    Args:
        *args (BackendArg): Positional args, first is input tensor.
        **kwargs (BackendArg): Additional args.

    Returns:
        EagerTensorProtocol: The reduced tensor.
    """
    import torch.distributed as dist

    tensor = _as_pt_tensor(args[0])
    if not dist.is_initialized():
        return tensor
    res: EagerTensorProtocol = adapt_backend_module(tensor).require_op("clone")()
    op_type = kwargs.get("op_type", "sum")
    dist.all_reduce(res, op=_get_torch_reduce_op(coerce_str(op_type)))
    return res


def _torch_reduce_scatter(*args: BackendArg, **kwargs: BackendArg) -> EagerTensorProtocol:
    """Execute torch distributed ReduceScatter.

    Args:
        *args (BackendArg): Positional args, first is input tensor.
        **kwargs (BackendArg): Additional args.

    Returns:
        EagerTensorProtocol: The reduce-scattered tensor.
    """
    import torch.distributed as dist

    torch_adp = adapt_backend_module(torch)
    tensor = _as_pt_tensor(args[0])
    if not dist.is_initialized():
        return tensor
    op_type = kwargs.get("op_type", "sum")
    axis = int(coerce_int(kwargs.get("axis", 0)))
    split_res = torch_adp.require_op("tensor_split")(tensor, int(coerce_int(dist.get_world_size())), dim=axis)
    tensor_list = list(coerce_operand_list(split_res))
    res: EagerTensorProtocol = torch_adp.require_op("empty_like")(tensor_list[0])
    dist.reduce_scatter(res, tensor_list, op=_get_torch_reduce_op(coerce_str(op_type)))
    return res


def _torch_all_to_all(*args: BackendArg, **kwargs: BackendArg) -> EagerTensorProtocol:
    """Execute torch distributed AllToAll.

    Args:
        *args (BackendArg): Positional args, first is input tensor.
        **kwargs (BackendArg): Additional args.

    Returns:
        EagerTensorProtocol: The scattered/gathered tensor.
    """
    import torch.distributed as dist

    torch_adp = adapt_backend_module(torch)
    tensor = _as_pt_tensor(args[0])
    if not dist.is_initialized():
        return tensor
    # Assuming input is pre-split along axis 0
    split_dim = int(coerce_int(kwargs.get("split_axis", 0)))
    concat_dim = int(coerce_int(kwargs.get("concat_axis", 0)))
    world_sz = int(coerce_int(dist.get_world_size()))
    split_res = torch_adp.require_op("tensor_split")(tensor, world_sz, dim=split_dim)
    input_tensor_list = list(coerce_operand_list(split_res))
    empty_like_fn = torch_adp.require_op("empty_like")
    output_tensor_list = [empty_like_fn(input_tensor_list[0]) for _ in range(world_sz)]
    dist.all_to_all(output_tensor_list, input_tensor_list)
    return torch_adp.require_op("cat")(output_tensor_list, dim=concat_dim)


def execute_op(cls: type | None, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager execute op operation on the active backend array.

    Args:
        cls (type | None): The class.
        op_type (str): Input `op_type` argument (op type).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        BackendNotSupportedError: If invalid inputs or unsupported runtime conditions trigger
            `BackendNotSupportedError`.
    """
    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings, resolve_target_api

    schema = load_backend_mappings("pytorch")
    if op_type in schema.operations and (
        schema.operations[op_type].target_api or schema.operations[op_type].custom_code
    ):
        import sys

        func = resolve_target_api(
            schema.operations[op_type].target_api, schema.operations[op_type].custom_code, sys.modules[__name__]
        )
        if func:
            return require_callable(func)(*args, **kwargs)
    custom_op_map = _get_custom_torch_op_map()
    if op_type in custom_op_map:
        return custom_op_map[op_type](*args, **kwargs)
    if adapt_backend_module(torch).has_symbol(op_type.lower()):
        func = adapt_backend_module(torch).require_symbol(op_type.lower())
        return func(*args, **kwargs)

    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    func = None
    for mod in [
        torch,
        F,
        adapt_backend_module(torch).get_symbol("linalg", None),
        adapt_backend_module(torch).get_symbol("fft", None),
    ]:
        if mod is not None and adapt_backend_module(mod).has_symbol(snake):
            func = adapt_backend_module(mod).require_symbol(snake)
            break
    if func is not None:
        return func(*args, **kwargs)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_func = global_eager_registry.get(op_type)
    if global_func is not None:
        return global_func(torch, *args, **kwargs)
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
