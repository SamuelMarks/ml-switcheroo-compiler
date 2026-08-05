# ruff: noqa: E501
"""Distributed ops eager."""

import threading

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: object, **kwargs: object) -> object:
    """Evaluate _np_axis_index operation.

    Args:
        backend_module (object): The backend_module parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return np.array(0)


class MockDistributedContext:
    """Mock Distributed Context."""

    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        """Init.

        Args:
            world_size (int): The world_size parameter.
            rank (int): The rank parameter.
        """
        self.world_size = world_size
        self.rank = rank
        self.mailboxes: dict[str, list[object]] = {}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(world_size) if world_size > 1 else None


_mock_dist_ctx = MockDistributedContext()


def set_mock_distributed_context(world_size: int, rank: int) -> None:
    """Set mock distributed context.

    Args:
        world_size (int): The world_size parameter.
        rank (int): The rank parameter.
    """
    global _mock_dist_ctx
    _mock_dist_ctx = MockDistributedContext(world_size, rank)


@numpy_eager_registry.register("AllReduce")
@numpy_eager_registry.register("NcclAllReduce")
@numpy_eager_registry.register("HierarchicalCopyAllReduce")
def _np_all_reduce(backend_module: object, tensor: object, op_type: str = "sum", *args: object, **kwargs: object) -> object:
    """Eager eval _np_all_reduce.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    # If world_size=1, just return the tensor
    if _mock_dist_ctx.world_size == 1:
        return backend_module.array(tensor)
    # Basic math simulation without actual cross-thread sync for simple tests
    # In a real mock, we would wait on the barrier and reduce across mailboxes.
    return backend_module.array(tensor)


@numpy_eager_registry.register("AllGather")
def _np_all_gather(backend_module: object, tensor: object, axis: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_all_gather.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    tensor = backend_module.array(tensor)
    if _mock_dist_ctx.world_size == 1:
        return backend_module.expand_dims(tensor, axis=axis) if axis is not None else tensor
    return backend_module.concatenate([tensor] * _mock_dist_ctx.world_size, axis=axis if axis is not None else 0)


@numpy_eager_registry.register("Broadcast")
def _np_broadcast(backend_module: object, tensor: object, root_rank: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_broadcast.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        root_rank (int): The root_rank parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    # For a mock simulation, we just return the tensor (assuming all ranks call this with the broadcasted value)
    return backend_module.array(tensor)


@numpy_eager_registry.register("ReduceScatter")
def _np_reduce_scatter(backend_module: object, tensor: object, op_type: str = "sum", axis: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_reduce_scatter.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        op_type (str): The op_type parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    if _mock_dist_ctx.world_size == 1:
        return backend_module.array(tensor)
    return backend_module.array(tensor)


@numpy_eager_registry.register("Reduce")
def _np_reduce(backend_module: object, tensor: object, root_rank: int = 0, op_type: str = "sum", *args: object, **kwargs: object) -> object:
    """Eager eval _np_reduce.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        root_rank (int): The root_rank parameter.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.array(tensor)


@numpy_eager_registry.register("ShardTensor")
def _np_shard_tensor(backend_module: object, tensor: object, *args: object, **kwargs: object) -> object:
    """Eager eval _np_shard_tensor.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.array(tensor)
