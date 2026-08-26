# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Distributed operations."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def shard_tensor(tensor: Tensor, device_mesh, layout):
    """Shard a tensor across devices.

    Args:
        tensor (Tensor): The tensor parameter.
        device_mesh (object): The device_mesh parameter.
        layout (object): The layout parameter.

    Returns:
        Tensor: Result.
    """
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

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def nccl_all_reduce(tensor: Tensor, op_type: str = "sum"):
    """NCCL all-reduce.

    Args:
        tensor (Tensor): The tensor parameter.
        op_type (str): The op_type parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("NcclAllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("NcclAllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("NcclAllReduce")
class NcclAllReduce(OpDef):
    """NCCL all-reduce op."""

    op_name = "NcclAllReduce"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def hierarchical_copy_all_reduce(tensor: Tensor, op_type: str = "sum"):
    """Hierarchical copy all-reduce.

    Args:
        tensor (Tensor): The tensor parameter.
        op_type (str): The op_type parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("HierarchicalCopyAllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("HierarchicalCopyAllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(OpDef):
    """Hierarchical copy all-reduce op."""

    op_name = "HierarchicalCopyAllReduce"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def broadcast(tensor: Tensor, root_rank: int = 0):
    """Broadcast.

    Args:
        tensor (Tensor): The tensor parameter.
        root_rank (int): The root_rank parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Broadcast", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), root_rank=root_rank)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("Broadcast", [tensor], {"root_rank": root_rank}, tensor.shape, tensor.dtype)


@register_op("Broadcast")
class Broadcast(OpDef):
    """Broadcast op."""

    op_name = "Broadcast"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def all_gather(tensor: Tensor, axis: int = 0):
    """All-gather.

    Args:
        tensor (Tensor): The tensor parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AllGather", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("AllGather", [tensor], {"axis": axis}, tensor.shape, tensor.dtype)  # simplified shape


@register_op("AllGather")
class AllGather(OpDef):
    """All-gather op."""

    op_name = "AllGather"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def reduce(tensor: Tensor, root_rank: int = 0, op_type: str = "sum"):
    """Reduce.

    Args:
        tensor (Tensor): The tensor parameter.
        root_rank (int): The root_rank parameter.
        op_type (str): The op_type parameter.

    Returns:
        Tensor: Result.
    """
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

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def all_reduce(tensor: Tensor, op_type: str = "sum"):
    """Provide generic SPMD AllReduce.

    Args:
        tensor (Tensor): The tensor parameter.
        op_type (str): The op_type parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AllReduce", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("AllReduce", [tensor], {"op_type": op_type}, tensor.shape, tensor.dtype)


@register_op("AllReduce")
class AllReduce(OpDef):
    """AllReduce op."""

    op_name = "AllReduce"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


def reduce_scatter(tensor: Tensor, op_type: str = "sum", axis: int = 0):
    """Provide generic SPMD ReduceScatter.

    Args:
        tensor (Tensor): The tensor parameter.
        op_type (str): The op_type parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ReduceScatter", (tensor.data if type(tensor).__name__ == "Tensor" else tensor), op_type=op_type, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", tensor.shape), tensor.dtype, tensor.device))
    return _emit_shape_node("ReduceScatter", [tensor], {"op_type": op_type, "axis": axis}, tensor.shape, tensor.dtype)


@register_op("ReduceScatter")
class ReduceScatter(OpDef):
    """ReduceScatter op."""

    op_name = "ReduceScatter"

    def infer_shape(self, tensor, **kwargs):
        """Infer shape.

        Args:
            tensor (object): The tensor parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(tensor, "shape", ())


@register_op("AllToAll")
class AllToAll(OpDef):
    """AllToAll operation."""

    op_name = "AllToAll"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("BroadcastArrays")
class BroadcastArrays(OpDef):
    """BroadcastArrays operation."""

    op_name = "BroadcastArrays"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """BroadcastTo operation."""

    op_name = "BroadcastTo"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if "shape" in kwargs:
            return kwargs["shape"]
        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("BroadcastToRank")
class BroadcastToRank(OpDef):
    """BroadcastToRank operation."""

    op_name = "BroadcastToRank"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        rank = kwargs.get("rank", 1)
        if args and hasattr(args[0], "shape"):
            shape = args[0].shape
            return (1,) * (rank - len(shape)) + shape if len(shape) < rank else shape
        return ()


@register_op("BroadcastedIota")
class BroadcastedIota(OpDef):
    """BroadcastedIota operation."""

    op_name = "BroadcastedIota"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("Pbroadcast")
class Pbroadcast(OpDef):
    """Pbroadcast operation."""

    op_name = "Pbroadcast"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def all_to_all(*args, **kwargs):
    """AllToAll frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("AllToAll", *args, **kwargs)
    return _emit_shape_node("AllToAll", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_arrays(*args, **kwargs):
    """BroadcastArrays frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastArrays", *args, **kwargs)
    return _emit_shape_node("BroadcastArrays", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_to(*args, **kwargs):
    """BroadcastTo frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastTo", *args, **kwargs)
    return _emit_shape_node("BroadcastTo", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcast_to_rank(*args, **kwargs):
    """BroadcastToRank frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastToRank", *args, **kwargs)
    return _emit_shape_node("BroadcastToRank", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def broadcasted_iota(*args, **kwargs):
    """BroadcastedIota frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("BroadcastedIota", *args, **kwargs)
    return _emit_shape_node("BroadcastedIota", list(args), kwargs, (), "int32")


def pbroadcast(*args, **kwargs):
    """Pbroadcast frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return get_active_backend().execute_op("Pbroadcast", *args, **kwargs)
    return _emit_shape_node("Pbroadcast", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


@register_op("Pmax")
class Pmax(OpDef):
    """Parallel maximum operator."""

    op_name = "Pmax"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("Pmin")
class Pmin(OpDef):
    """Parallel minimum operator."""

    op_name = "Pmin"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("Outfeed")
class Outfeed(OpDef):
    """Write to the outfeed queue."""

    op_name = "Outfeed"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Pshuffle")
class Pshuffle(OpDef):
    """Parallel shuffle operator."""

    op_name = "Pshuffle"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("Pswapaxes")
class Pswapaxes(OpDef):
    """Parallel swapaxes operator."""

    op_name = "Pswapaxes"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        axis = kwargs.get("axis", args[2] if len(args) > 2 else 0)
        shape = list(getattr(x, "shape", ()))
        if shape and axis < len(shape):
            shape[axis] = None
        return tuple(shape)


@register_op("Ppermute")
class Ppermute(OpDef):
    """Parallel permute operator."""

    op_name = "Ppermute"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("PsumScatter")
class PsumScatter(OpDef):
    """Parallel sum scatter operator."""

    op_name = "PsumScatter"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else None
        scatter_dimension = kwargs.get("scatter_dimension", 0)
        shape = list(getattr(x, "shape", ()))
        if shape and scatter_dimension < len(shape):
            shape[scatter_dimension] = None
        return tuple(shape)


def outfeed(*args, **kwargs):
    """Write to the outfeed queue.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Outfeed", *args, **kwargs)


def pmax(*args, **kwargs):
    """Evaluate pmax operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pmax", *args, **kwargs)


def pmin(*args, **kwargs):
    """Evaluate pmin operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pmin", *args, **kwargs)


def ppermute(*args, **kwargs):
    """Permute data across the mapped axis.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ppermute", *args, **kwargs)


def pshuffle(*args, **kwargs):
    """Shuffle data across the mapped axis.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pshuffle", *args, **kwargs)


def psum_scatter(*args, **kwargs):
    """Scatter sum across a mapped axis.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PsumScatter", *args, **kwargs)


def pswapaxes(*args, **kwargs):
    """Swap axes of the data.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pswapaxes", *args, **kwargs)


@register_op("Send")
class Send(OpDef):
    """Point-to-Point Send operation for Pipeline Parallelism."""

    op_name = "Send"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: Empty tuple as Send does not return tensor data.
        """
        return ()


@register_op("Recv")
class Recv(OpDef):
    """Point-to-Point Recv operation for Pipeline Parallelism."""

    op_name = "Recv"

    def infer_shape(self, *args, **kwargs):
        """Infer shape based on expected kwargs.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Tensor: Expected received tensor shape.
        """
        return kwargs.get("shape", ())


def send(tensor, dst_rank: int, tag: int = 0):
    """Send tensor to destination rank."""
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    return TracingNodeBuilder.emit_tracing_node("Send", tensor, dst_rank=dst_rank, tag=tag)


def recv(shape, dtype: str, src_rank: int, tag: int = 0):
    """Receive tensor from source rank."""
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    return TracingNodeBuilder.emit_tracing_node("Recv", shape=shape, dtype=dtype, src_rank=src_rank, tag=tag)
