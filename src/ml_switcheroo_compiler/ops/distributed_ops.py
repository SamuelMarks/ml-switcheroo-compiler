# ruff: noqa: D102, ANN401
"""Distributed operations."""

from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def shard_tensor(tensor: Tensor, device_mesh: Any, layout: Any) -> Tensor:
    """Shard a tensor across devices."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "ShardTensor",
            (tensor.data if type(tensor).__name__ == "Tensor" else tensor),
            device_mesh=device_mesh,
            layout=layout,
        )
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("ShardTensor", [tensor], {"device_mesh": device_mesh, "layout": layout}, tensor.shape, tensor.dtype)


@register_op("ShardTensor")
class ShardTensor(OpDef):
    """Shard tensor op."""

    op_name = "ShardTensor"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def nccl_all_reduce(tensor: Tensor, op_type: str = "sum") -> Tensor:
    """NCCL all-reduce."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("NcclAllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("NcclAllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("NcclAllReduce")
class NcclAllReduce(OpDef):
    """NCCL all-reduce op."""

    op_name = "NcclAllReduce"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def hierarchical_copy_all_reduce(tensor: Tensor, op_type: str = "sum") -> Tensor:
    """Hierarchical copy all-reduce."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("HierarchicalCopyAllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("HierarchicalCopyAllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(OpDef):
    """Hierarchical copy all-reduce op."""

    op_name = "HierarchicalCopyAllReduce"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def broadcast(tensor: Tensor, root_rank: int = 0) -> Tensor:
    """Broadcast."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Broadcast", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), root_rank=root_rank)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("Broadcast", [tensor], {"root_rank": root_rank}, tensor.shape, tensor.dtype)


@register_op("Broadcast")
class Broadcast(OpDef):
    """Broadcast op."""

    op_name = "Broadcast"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def all_gather(tensor: Tensor, axis: int = 0) -> Tensor:
    """All-gather."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AllGather", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("AllGather", [tensor], {"axis": axis}, tensor.shape, tensor.dtype)  # simplified shape


@register_op("AllGather")
class AllGather(OpDef):
    """All-gather op."""

    op_name = "AllGather"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def reduce(tensor: Tensor, root_rank: int = 0, op_type: str = "sum") -> Tensor:
    """Reduce."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Reduce",
            (tensor.data if type(tensor).__name__ == "Tensor" else tensor),
            root_rank=root_rank,
            op_type=op_type,
        )
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("Reduce", [tensor], {"root_rank": root_rank, "op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("Reduce")
class Reduce(OpDef):
    """Reduce op."""

    op_name = "Reduce"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def all_reduce(tensor: Tensor, op_type: str = "sum") -> Tensor:
    """Generic SPMD AllReduce."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("AllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("AllReduce")
class AllReduce(OpDef):
    """AllReduce op."""

    op_name = "AllReduce"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


def reduce_scatter(tensor: Tensor, op_type: str = "sum", axis: int = 0) -> Tensor:
    """Generic SPMD ReduceScatter."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ReduceScatter", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("ReduceScatter", [tensor], {"op_type": op_type, "axis": axis}, tensor.shape, tensor.dtype)


@register_op("ReduceScatter")
class ReduceScatter(OpDef):
    """ReduceScatter op."""

    op_name = "ReduceScatter"

    def infer_shape(self, tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("AllToAll")
class AllToAll(OpDef):
    """AllToAll operation."""

    op_name = "AllToAll"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("BroadcastArrays")
class BroadcastArrays(OpDef):
    """BroadcastArrays operation."""

    op_name = "BroadcastArrays"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """BroadcastTo operation."""

    op_name = "BroadcastTo"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("BroadcastToRank")
class BroadcastToRank(OpDef):
    """BroadcastToRank operation."""

    op_name = "BroadcastToRank"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("BroadcastedIota")
class BroadcastedIota(OpDef):
    """BroadcastedIota operation."""

    op_name = "BroadcastedIota"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Pbroadcast")
class Pbroadcast(OpDef):
    """Pbroadcast operation."""

    op_name = "Pbroadcast"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


def all_to_all(*args: object, **kwargs: object) -> object:
    """AllToAll frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("AllToAll", *args, **kwargs)
    return _emit_shape_node("AllToAll", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_arrays(*args: object, **kwargs: object) -> object:
    """BroadcastArrays frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastArrays", *args, **kwargs)
    return _emit_shape_node("BroadcastArrays", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_to(*args: object, **kwargs: object) -> object:
    """BroadcastTo frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastTo", *args, **kwargs)
    return _emit_shape_node("BroadcastTo", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_to_rank(*args: object, **kwargs: object) -> object:
    """BroadcastToRank frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastToRank", *args, **kwargs)
    return _emit_shape_node("BroadcastToRank", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcasted_iota(*args: object, **kwargs: object) -> object:
    """BroadcastedIota frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastedIota", *args, **kwargs)
    return _emit_shape_node("BroadcastedIota", list(args), kwargs, (), "int32")


def pbroadcast(*args: object, **kwargs: object) -> object:
    """Pbroadcast frontend."""
    if config.eager_mode:
        return get_active_backend().execute_op("Pbroadcast", *args, **kwargs)
    return _emit_shape_node("Pbroadcast", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


@register_op("Pmax")
class Pmax(OpDef):
    """Parallel maximum operator."""

    op_name = "Pmax"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Pmin")
class Pmin(OpDef):
    """Parallel minimum operator."""

    op_name = "Pmin"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Outfeed")
class Outfeed(OpDef):
    """Write to the outfeed queue."""

    op_name = "Outfeed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Pshuffle")
class Pshuffle(OpDef):
    """Parallel shuffle operator."""

    op_name = "Pshuffle"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Pswapaxes")
class Pswapaxes(OpDef):
    """Parallel swapaxes operator."""

    op_name = "Pswapaxes"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Ppermute")
class Ppermute(OpDef):
    """Parallel permute operator."""

    op_name = "Ppermute"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("PsumScatter")
class PsumScatter(OpDef):
    """Parallel sum scatter operator."""

    op_name = "PsumScatter"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()
