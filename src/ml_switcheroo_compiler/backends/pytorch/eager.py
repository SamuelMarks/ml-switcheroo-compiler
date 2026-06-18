"""Backend utilities."""


def _execute_tensor_scatter_max(*args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]  # type: ignore
    flat_idx = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))
    res = tensor.clone().flatten()
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amax", include_self=True)
    return res.reshape(tensor.shape)


def _execute_tensor_scatter_min(*args: object, **kwargs: object) -> object:
    tensor, indices, updates = args[0], args[1], args[2]  # type: ignore
    flat_idx = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))
    res = tensor.clone().flatten()
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amin", include_self=True)
    return res.reshape(tensor.shape)


def _execute_tensor_scatter_update(*args: object, **kwargs: object) -> object:
    return args[0].clone().index_put_(tuple(args[1].unbind(-1)), args[2])  # type: ignore


def _execute_tensor_scatter_add(*args: object, **kwargs: object) -> object:
    return args[0].clone().index_put_(tuple(args[1].unbind(-1)), args[2], accumulate=True)  # type: ignore


def _execute_power_iteration(*args: object, **kwargs: object) -> object:
    import torch

    w = args[0]
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = torch.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype, device=w.device)
    for _ in range(num_iters):
        w_t = w.transpose(-1, -2)
        v = torch.matmul(w_t, u)
        v = v / (torch.linalg.norm(v, dim=-2, keepdim=True) + 1e-12)
        u = torch.matmul(w, v)
        u = u / (torch.linalg.norm(u, dim=-2, keepdim=True) + 1e-12)
    sigma = torch.matmul(u.transpose(-1, -2), torch.matmul(w, v))
    return (v.squeeze(-1), u.squeeze(-1), sigma.squeeze(-1).squeeze(-1))


def _execute_broadcast_to(*args: object, **kwargs: object) -> object:
    return args[0].expand(kwargs["shape"])  # type: ignore


def _get_custom_torch_op_map() -> dict:
    return {
        "TensorScatterUpdate": _execute_tensor_scatter_update,
        "TensorScatterAdd": _execute_tensor_scatter_add,
        "TensorScatterMax": _execute_tensor_scatter_max,
        "TensorScatterMin": _execute_tensor_scatter_min,
        "PowerIteration": _execute_power_iteration,
        "BroadcastTo": _execute_broadcast_to,
    }


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
    import torch

    op_map = {
        "Add": torch.add,
        "Subtract": torch.sub,
        "Multiply": torch.mul,
        "TrueDivide": torch.div,
        "Exp": torch.exp,
        "Log": torch.log,
        "Matmul": torch.matmul,
        "Sin": torch.sin,
        "Cos": torch.cos,
        "Sum": torch.sum,
        "Mean": torch.mean,
        "Max": torch.max,
        "Min": torch.min,
        "Reshape": torch.reshape,
        "Transpose": torch.transpose,
        "Equal": torch.eq,
        "NotEqual": torch.ne,
        "Greater": torch.gt,
        "Less": torch.lt,
        "Negative": torch.neg,
    }

    if op_type in op_map:
        return op_map[op_type](*args, **kwargs)

    custom_op_map = _get_custom_torch_op_map()
    if op_type in custom_op_map:
        return custom_op_map[op_type](*args, **kwargs)

    try:
        func = getattr(torch, op_type.lower())
        return func(*args, **kwargs)
    except AttributeError:
        msg = f"Operation '{op_type}' is not supported by torch backend."
        raise NotImplementedError(msg) from None
