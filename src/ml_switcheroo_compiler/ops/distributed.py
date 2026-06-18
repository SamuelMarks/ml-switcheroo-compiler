"""Distributed execution operations."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def shard_tensor(tensor: Tensor, device_mesh: object, layout: object) -> Tensor:
    """Physically partitions a tensor across a device mesh according to a layout.

    Args:
        tensor (Tensor): The input tensor.
        device_mesh (object): The DeviceMesh.
        layout (object): The layout mapping.

    Returns:
        Tensor: The sharded tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ShardTensor", tensor.data, device_mesh=device_mesh, layout=layout
        )
        return Tensor(backend.array(data), backend.array(data).shape, tensor.dtype, tensor.device)
    return _emit_shape_node(
        "ShardTensor", [tensor], {"device_mesh": device_mesh, "layout": layout}, (), tensor.dtype
    )


def all_reduce(tensor: Tensor, op: str = "sum") -> Tensor:
    """Performs an all-reduce operation across devices.

    Args:
        tensor (Tensor): The input tensor.
        op (str): The reduction operation (e.g., 'sum', 'mean').

    Returns:
        Tensor: The reduced tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AllReduce", tensor.data, op=op)
        return Tensor(backend.array(data), backend.array(data).shape, tensor.dtype, tensor.device)
    return _emit_shape_node("AllReduce", [tensor], {"op": op}, (), tensor.dtype)


def all_gather(tensor: Tensor, axis: int = 0) -> Tensor:
    """Gathers a tensor from all devices.

    Args:
        tensor (Tensor): The input tensor.
        axis (int): The axis along which to concatenate.

    Returns:
        Tensor: The gathered tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AllGather", tensor.data, axis=axis)
        return Tensor(backend.array(data), backend.array(data).shape, tensor.dtype, tensor.device)
    return _emit_shape_node("AllGather", [tensor], {"axis": axis}, (), tensor.dtype)


def reduce_scatter(tensor: Tensor, op: str = "sum", axis: int = 0) -> Tensor:
    """Performs a reduction followed by a scatter across devices.

    Args:
        tensor (Tensor): The input tensor.
        op (str): The reduction operation.
        axis (int): The scatter axis.

    Returns:
        Tensor: The scattered tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("ReduceScatter", tensor.data, op=op, axis=axis)
        return Tensor(backend.array(data), backend.array(data).shape, tensor.dtype, tensor.device)
    return _emit_shape_node("ReduceScatter", [tensor], {"op": op, "axis": axis}, (), tensor.dtype)


@register_op("ShardTensor")
class ShardTensorOp(OpDef):
    """ShardTensor op."""

    def infer_shape(self, tensor: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AllReduce")
class AllReduceOp(OpDef):
    """AllReduce op."""

    def infer_shape(self, tensor: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AllGather")
class AllGatherOp(OpDef):
    """AllGather op."""

    def infer_shape(self, tensor: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ReduceScatter")
class ReduceScatterOp(OpDef):
    """ReduceScatter op."""

    def infer_shape(self, tensor: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()
