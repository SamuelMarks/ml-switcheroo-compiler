"""Native PyTorch distributed collective dispatch bindings."""

from __future__ import annotations

import logging

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_int, require_callable


def pytorch_all_reduce(
    tensor: BackendArg,
    op: str = "SUM",
    group: BackendArg | None = None,
) -> BackendArg:
    """Execute native PyTorch all_reduce on a tensor.

    Args:
        tensor (BackendArg): Torch tensor or array.
        op (str): Reduction operator ("SUM", "PRODUCT", "MAX", "MIN").
        group (BackendArg | None): Process group handle.

    Returns:
        BackendArg: In-place or returned reduced tensor.
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

            t = tensor if isinstance(tensor, torch.Tensor) else require_callable(torch.as_tensor)(tensor)
            dist.all_reduce(t, op=reduce_op, group=group)
            return t
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_all_gather(
    tensor: BackendArg,
    axis: int = 0,
    group: BackendArg | None = None,
) -> BackendArg:
    """Execute native PyTorch all_gather on a tensor.

    Args:
        tensor (BackendArg): Torch tensor or array.
        axis (int): Concatenation axis.
        group (BackendArg | None): Process group handle.

    Returns:
        BackendArg: Concatenated global tensor across all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist_mod = dist
            world_size = int(coerce_int(dist_mod.get_world_size(group)))
            t = tensor if isinstance(tensor, torch.Tensor) else require_callable(torch.as_tensor)(tensor)
            gather_list = [require_callable(torch.empty_like)(t) for _ in range(world_size)]
            dist_mod.all_gather(gather_list, t, group=group)
            return require_callable(torch.cat)(gather_list, dim=axis)
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_reduce_scatter(
    tensor: BackendArg,
    op: str = "SUM",
    scatter_dim: int = 0,
    group: BackendArg | None = None,
) -> BackendArg:
    """Execute native PyTorch reduce_scatter on a tensor.

    Args:
        tensor (BackendArg): Torch tensor or array.
        op (str): Reduction operator.
        scatter_dim (int): Dimension to scatter across ranks.
        group (BackendArg | None): Process group handle.

    Returns:
        BackendArg: Reduced local slice.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist_mod = dist
            world_size = int(coerce_int(dist_mod.get_world_size(group)))
            t = tensor if isinstance(tensor, torch.Tensor) else require_callable(torch.as_tensor)(tensor)
            input_list = list(torch.chunk(t, world_size, dim=scatter_dim))
            output = require_callable(torch.empty_like)(input_list[0])
            reduce_op = dist.ReduceOp.SUM
            if op.upper() in ("PROD", "PRODUCT"):
                reduce_op = dist.ReduceOp.PRODUCT
            dist_mod.reduce_scatter(output, input_list, op=reduce_op, group=group)
            return output
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor


def pytorch_broadcast(
    tensor: BackendArg,
    src: int = 0,
    group: BackendArg | None = None,
) -> BackendArg:
    """Execute native PyTorch broadcast on a tensor.

    Args:
        tensor (BackendArg): Torch tensor or array.
        src (int): Source root rank.
        group (BackendArg | None): Process group handle.

    Returns:
        BackendArg: Broadcasted tensor on all ranks.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            t = tensor if isinstance(tensor, torch.Tensor) else require_callable(torch.as_tensor)(tensor)
            dist.broadcast(t, src=src, group=group)
            return t
    except ImportError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return tensor
