"""Native PyTorch distributed collective dispatch bindings."""

from __future__ import annotations

import logging

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, ensure_int
from ml_switcheroo_compiler.backends.protocols import ensure_mod


def pytorch_all_reduce(
    tensor: EagerValue,
    op: str = "SUM",
    group: EagerValue | None = None,
) -> EagerValue:
    """Execute native PyTorch all_reduce on a tensor.

    Args:
        tensor (EagerValue): Torch tensor or array.
        op (str): Reduction operator ("SUM", "PRODUCT", "MAX", "MIN").
        group (Any | None): Process group handle.

    Returns:
        EagerValue: In-place or returned reduced tensor.
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
            ensure_mod(dist).all_reduce(t, op=reduce_op, group=group)
            return ensure_eager_tensor(t)
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_all_gather(
    tensor: EagerValue,
    axis: int = 0,
    group: EagerValue | None = None,
) -> EagerValue:
    """Execute native PyTorch all_gather on a tensor.

    Args:
        tensor (EagerValue): Torch tensor or array.
        axis (int): Concatenation axis.
        group (Any | None): Process group handle.

    Returns:
        EagerValue: Concatenated global tensor across all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist_mod = ensure_mod(dist)
            world_size = int(ensure_int(dist_mod.get_world_size(group)))
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            gather_list = [torch.empty_like(t) for _ in range(world_size)]
            dist_mod.all_gather(gather_list, t, group=group)
            return ensure_eager_tensor(torch.cat(gather_list, dim=axis))
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_reduce_scatter(
    tensor: EagerValue,
    op: str = "SUM",
    scatter_dim: int = 0,
    group: EagerValue | None = None,
) -> EagerValue:
    """Execute native PyTorch reduce_scatter on a tensor.

    Args:
        tensor (EagerValue): Torch tensor or array.
        op (str): Reduction operator.
        scatter_dim (int): Dimension to scatter across ranks.
        group (Any | None): Process group handle.

    Returns:
        EagerValue: Reduced local slice.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist_mod = ensure_mod(dist)
            world_size = int(ensure_int(dist_mod.get_world_size(group)))
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            input_list = list(torch.chunk(t, world_size, dim=scatter_dim))
            output = torch.empty_like(input_list[0])
            reduce_op = dist.ReduceOp.SUM
            if op.upper() in ("PROD", "PRODUCT"):
                reduce_op = dist.ReduceOp.PRODUCT
            dist_mod.reduce_scatter(output, input_list, op=reduce_op, group=group)
            return ensure_eager_tensor(output)
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_broadcast(
    tensor: EagerValue,
    src: int = 0,
    group: EagerValue | None = None,
) -> EagerValue:
    """Execute native PyTorch broadcast on a tensor.

    Args:
        tensor (EagerValue): Torch tensor or array.
        src (int): Source root rank.
        group (Any | None): Process group handle.

    Returns:
        EagerValue: Broadcasted tensor on all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            ensure_mod(dist).broadcast(t, src=src, group=group)
            return ensure_eager_tensor(t)
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor
