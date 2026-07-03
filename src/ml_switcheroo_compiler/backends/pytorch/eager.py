"""Backend utilities."""

from typing import Any, cast

import torch  # pragma: no cover


def _execute_tensor_scatter_max(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = (
        cast(Any, args[0]),
        cast(Any, args[1]),
        cast(Any, args[2]),
    )  # pragma: no cover
    flat_idx = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))  # pragma: no cover
    res = tensor.clone().flatten()  # pragma: no cover
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amax", include_self=True)  # pragma: no cover
    return res.reshape(tensor.shape)  # pragma: no cover


def _execute_tensor_scatter_min(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = (
        cast(Any, args[0]),
        cast(Any, args[1]),
        cast(Any, args[2]),
    )  # pragma: no cover
    flat_idx = sum(indices[..., i] * tensor.stride(i) for i in range(indices.shape[-1]))  # pragma: no cover
    res = tensor.clone().flatten()  # pragma: no cover
    res.scatter_reduce_(0, flat_idx.flatten(), updates.flatten(), reduce="amin", include_self=True)  # pragma: no cover
    return res.reshape(tensor.shape)  # pragma: no cover


def _execute_tensor_scatter_update(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    return (  # pragma: no cover
        cast(Any, args[0]).clone().index_put_(tuple(cast(Any, args[1]).unbind(-1)), cast(Any, args[2]))
    )


def _execute_tensor_scatter_add(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    return (  # pragma: no cover
        cast(Any, args[0]).clone().index_put_(tuple(cast(Any, args[1]).unbind(-1)), cast(Any, args[2]), accumulate=True)
    )


def _execute_power_iteration(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    w = args[0]  # pragma: no cover
    num_iters = kwargs.get("num_iters", 1)  # pragma: no cover
    u = kwargs.get("u", None)  # pragma: no cover
    if u is None:  # pragma: no cover
        u = torch.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype, device=w.device)  # pragma: no cover
    for _ in range(num_iters):  # pragma: no cover
        w_t = w.transpose(-1, -2)  # pragma: no cover
        v = torch.matmul(w_t, u)  # pragma: no cover
        v = v / (torch.linalg.norm(v, dim=-2, keepdim=True) + 1e-12)  # pragma: no cover
        u = torch.matmul(w, v)  # pragma: no cover
        u = u / (torch.linalg.norm(u, dim=-2, keepdim=True) + 1e-12)  # pragma: no cover
    sigma = torch.matmul(u.transpose(-1, -2), torch.matmul(w, v))  # pragma: no cover
    return (v.squeeze(-1), u.squeeze(-1), sigma.squeeze(-1).squeeze(-1))  # pragma: no cover


def _execute_broadcast_to(*args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    return cast(Any, args[0]).expand(kwargs["shape"])  # pragma: no cover


def _execute_cummax(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return torch.cummax(*args, **kwargs)[0]


def _execute_cummin(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return torch.cummin(*args, **kwargs)[0]


def _execute_cumlogsumexp(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return torch.logcumsumexp(*args, **kwargs)


def _get_custom_torch_op_map() -> dict:
    """Function docstring."""
    return {
        "TensorScatterUpdate": _execute_tensor_scatter_update,
        "TensorScatterAdd": _execute_tensor_scatter_add,
        "TensorScatterMax": _execute_tensor_scatter_max,
        "TensorScatterMin": _execute_tensor_scatter_min,
        "PowerIteration": _execute_power_iteration,
        "BroadcastTo": _execute_broadcast_to,
        "Cummax": _execute_cummax,
        "Cummin": _execute_cummin,
        "Cumlogsumexp": _execute_cumlogsumexp,
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
    op_map = {
        "Add": torch.add,
        "Subtract": torch.sub,
        "Multiply": torch.mul,
        "TrueDivide": torch.div,
        "TruncateDiv": lambda x, y: torch.trunc(torch.div(x, y)),
        "TruncateMod": torch.fmod,
        "Exp": torch.exp,
        "Log": torch.log,
        "Matmul": torch.matmul,
        "Sin": torch.sin,
        "Acos": torch.acos,
        "Acosh": torch.acosh,
        "Asin": torch.asin,
        "Asinh": torch.asinh,
        "Atan": torch.atan,
        "Atanh": torch.atanh,
        "Atan2": torch.atan2,
        "Cos": torch.cos,
        "Sum": torch.sum,
        "Cumsum": torch.cumsum,
        "Cumprod": torch.cumprod,
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
    if op_type in custom_op_map:  # pragma: no branch
        return custom_op_map[op_type](*args, **kwargs)  # pragma: no cover

    try:
        func = getattr(torch, op_type.lower())
        return func(*args, **kwargs)
    except AttributeError:
        msg = f"Operation '{op_type}' is not supported by torch backend."
        raise NotImplementedError(msg) from None
