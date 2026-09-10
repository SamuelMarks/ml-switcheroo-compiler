# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Native PyTorch distributed collective dispatch bindings."""

from __future__ import annotations

from typing import Any


def pytorch_all_reduce(
    tensor: Any,
    op: str = "SUM",
    group: Any | None = None,
) -> Any:
    """Execute native PyTorch all_reduce on a tensor.

    Args:
        tensor (Any): Torch tensor or array.
        op (str): Reduction operator ("SUM", "PRODUCT", "MAX", "MIN").
        group (Any | None): Process group handle.

    Returns:
        Any: In-place or returned reduced tensor.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            reduce_op = dist.ReduceOp.SUM
            if op.upper() in ("PROD", "PRODUCT"):
                reduce_op = dist.ReduceOp.PRODUCT
            elif op.upper() == "MAX":
                reduce_op = dist.ReduceOp.MAX
            elif op.upper() == "MIN":
                reduce_op = dist.ReduceOp.MIN

            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            dist.all_reduce(t, op=reduce_op, group=group)
            return t
    except ImportError:
        pass

    return tensor


def pytorch_all_gather(
    tensor: Any,
    axis: int = 0,
    group: Any | None = None,
) -> Any:
    """Execute native PyTorch all_gather on a tensor.

    Args:
        tensor (Any): Torch tensor or array.
        axis (int): Concatenation axis.
        group (Any | None): Process group handle.

    Returns:
        Any: Concatenated global tensor across all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group)
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            gather_list = [torch.empty_like(t) for _ in range(world_size)]
            dist.all_gather(gather_list, t, group=group)
            return torch.cat(gather_list, dim=axis)
    except ImportError:
        pass

    return tensor


def pytorch_reduce_scatter(
    tensor: Any,
    op: str = "SUM",
    scatter_dim: int = 0,
    group: Any | None = None,
) -> Any:
    """Execute native PyTorch reduce_scatter on a tensor.

    Args:
        tensor (Any): Torch tensor or array.
        op (str): Reduction operator.
        scatter_dim (int): Dimension to scatter across ranks.
        group (Any | None): Process group handle.

    Returns:
        Any: Reduced local slice.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group)
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            input_list = list(torch.chunk(t, world_size, dim=scatter_dim))
            output = torch.empty_like(input_list[0])
            reduce_op = dist.ReduceOp.SUM
            if op.upper() in ("PROD", "PRODUCT"):
                reduce_op = dist.ReduceOp.PRODUCT
            dist.reduce_scatter(output, input_list, op=reduce_op, group=group)
            return output
    except ImportError:
        pass

    return tensor


def pytorch_broadcast(
    tensor: Any,
    src: int = 0,
    group: Any | None = None,
) -> Any:
    """Execute native PyTorch broadcast on a tensor.

    Args:
        tensor (Any): Torch tensor or array.
        src (int): Source root rank.
        group (Any | None): Process group handle.

    Returns:
        Any: Broadcasted tensor on all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            dist.broadcast(t, src=src, group=group)
            return t
    except ImportError:
        pass

    return tensor
