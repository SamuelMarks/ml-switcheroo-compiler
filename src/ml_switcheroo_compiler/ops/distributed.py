"""Distributed execution operations."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


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
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
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
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
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
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
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
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
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


def device_put_replicated(tensor: Tensor, devices: object) -> Tensor:
    """Replicates a tensor across multiple devices.

    Args:
        tensor (Tensor): The input tensor.
        devices (object): The devices to replicate to.

    Returns:
        Tensor: The replicated tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "DevicePutReplicated", tensor.data, devices=devices
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(
        "DevicePutReplicated", [tensor], {"devices": devices}, (), tensor.dtype
    )  # pragma: no cover


def device_put_sharded(tensors: list[Tensor], devices: object) -> Tensor:
    """Puts a list of shards onto devices.

    Args:
        tensors (list[Tensor]): The input tensor shards.
        devices (object): The devices to put to.

    Returns:
        Tensor: The sharded tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data_list = [t.data for t in tensors]  # pragma: no cover
        data = backend.execute_op(
            "DevicePutSharded", data_list, devices=devices
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensors[0].dtype, tensors[0].device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(
        "DevicePutSharded", tensors, {"devices": devices}, (), tensors[0].dtype
    )  # pragma: no cover


def all_to_all(tensor: Tensor, split_axis: int, concat_axis: int, axis_name: str) -> Tensor:
    """All-to-all scatter-gather operation.

    Args:
        tensor (Tensor): Input tensor.
        split_axis (int): Axis to split.
        concat_axis (int): Axis to concat.
        axis_name (str): The axis name.

    Returns:
        Tensor: The output tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(  # pragma: no cover
            "AllToAll",  # pragma: no cover
            tensor.data,  # pragma: no cover
            split_axis=split_axis,  # pragma: no cover
            concat_axis=concat_axis,  # pragma: no cover
            axis_name=axis_name,  # pragma: no cover
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(  # pragma: no cover
        "AllToAll",
        [tensor],
        {"split_axis": split_axis, "concat_axis": concat_axis, "axis_name": axis_name},
        (),
        tensor.dtype,
    )


def pmax(tensor: Tensor, axis_name: str) -> Tensor:
    """Parallel max over axis.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): The axis name.

    Returns:
        Tensor: The output tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("Pmax", tensor.data, axis_name=axis_name)  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(
        "Pmax", [tensor], {"axis_name": axis_name}, (), tensor.dtype
    )  # pragma: no cover


def pmin(tensor: Tensor, axis_name: str) -> Tensor:
    """Parallel min over axis.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): The axis name.

    Returns:
        Tensor: The output tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("Pmin", tensor.data, axis_name=axis_name)  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(
        "Pmin", [tensor], {"axis_name": axis_name}, (), tensor.dtype
    )  # pragma: no cover


def psum_scatter(tensor: Tensor, scatter_dimension: int, axis_name: str) -> Tensor:
    """Parallel sum and scatter over axis.

    Args:
        tensor (Tensor): Input tensor.
        scatter_dimension (int): Scatter dim.
        axis_name (str): The axis name.

    Returns:
        Tensor: The output tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(  # pragma: no cover
            "PsumScatter",
            tensor.data,
            scatter_dimension=scatter_dimension,
            axis_name=axis_name,  # pragma: no cover
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(  # pragma: no cover
        "PsumScatter",
        [tensor],
        {"scatter_dimension": scatter_dimension, "axis_name": axis_name},
        (),
        tensor.dtype,
    )


def pswapaxes(tensor: Tensor, axis_name: str, axis: int) -> Tensor:
    """Parallel swap axes.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): The axis name.
        axis (int): The local axis to swap with.

    Returns:
        Tensor: The output tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "Pswapaxes", tensor.data, axis_name=axis_name, axis=axis
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),  # pragma: no cover
            TensorConfig(
                backend.array(data).shape, tensor.dtype, tensor.device
            ),  # pragma: no cover
        )  # pragma: no cover
    return _emit_shape_node(  # pragma: no cover
        "Pswapaxes", [tensor], {"axis_name": axis_name, "axis": axis}, (), tensor.dtype
    )


@register_op("DevicePutReplicated")
class DevicePutReplicated(OpDef):
    """DevicePutReplicated op."""

    op_name = "DevicePutReplicated"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("DevicePutSharded")
class DevicePutSharded(OpDef):
    """DevicePutSharded op."""

    op_name = "DevicePutSharded"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("AllToAll")
class AllToAll(OpDef):
    """AllToAll op."""

    op_name = "AllToAll"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("Pmax")
class Pmax(OpDef):
    """Pmax op."""

    op_name = "Pmax"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("Pmin")
class Pmin(OpDef):
    """Pmin op."""

    op_name = "Pmin"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("PsumScatter")
class PsumScatter(OpDef):
    """PsumScatter op."""

    op_name = "PsumScatter"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


@register_op("Pswapaxes")
class Pswapaxes(OpDef):
    """Pswapaxes op."""

    op_name = "Pswapaxes"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        """Infer shape."""
        return ()


def pbroadcast(tensor: Tensor, axis_name: str) -> Tensor:
    """Parallel broadcast.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): Axis name.

    Returns:
        Tensor: Broadcasted tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pbroadcast", tensor.data, axis_name=axis_name)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    return _emit_shape_node("Pbroadcast", [tensor], {"axis_name": axis_name}, (), tensor.dtype)


def pdot(lhs: Tensor, rhs: Tensor, axis_name: str) -> Tensor:
    """Parallel dot.

    Args:
        lhs (Tensor): Left hand side tensor.
        rhs (Tensor): Right hand side tensor.
        axis_name (str): Axis name.

    Returns:
        Tensor: Resulting tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pdot", lhs.data, rhs.data, axis_name=axis_name)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, lhs.dtype, lhs.device)
        )
    return _emit_shape_node("Pdot", [lhs, rhs], {"axis_name": axis_name}, (), lhs.dtype)


def ppermute(tensor: Tensor, axis_name: str, perm: object) -> Tensor:
    """Parallel permute.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): Axis name.
        perm (object): Permutation.

    Returns:
        Tensor: Permuted tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ppermute", tensor.data, axis_name=axis_name, perm=perm)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    return _emit_shape_node(
        "Ppermute", [tensor], {"axis_name": axis_name, "perm": perm}, (), tensor.dtype
    )


def pshuffle(tensor: Tensor, axis_name: str, perm: object) -> Tensor:
    """Parallel shuffle.

    Args:
        tensor (Tensor): Input tensor.
        axis_name (str): Axis name.
        perm (object): Permutation.

    Returns:
        Tensor: Shuffled tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pshuffle", tensor.data, axis_name=axis_name, perm=perm)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    return _emit_shape_node(
        "Pshuffle", [tensor], {"axis_name": axis_name, "perm": perm}, (), tensor.dtype
    )


@register_op("Pbroadcast")
class Pbroadcast(OpDef):
    """Pbroadcast op."""

    op_name = "Pbroadcast"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return getattr(args[0], "shape", ()) if args else ()


@register_op("Pdot")
class Pdot(OpDef):
    """Pdot op."""

    op_name = "Pdot"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Ppermute")
class Ppermute(OpDef):
    """Ppermute op."""

    op_name = "Ppermute"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return getattr(args[0], "shape", ()) if args else ()


@register_op("Pshuffle")
class Pshuffle(OpDef):
    """Pshuffle op."""

    op_name = "Pshuffle"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return getattr(args[0], "shape", ()) if args else ()


@register_op("Infeed")
class Infeed(OpDef):
    """Infeed operator."""

    op_name = "Infeed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return kwargs.get("shape", ())


@register_op("Outfeed")
class Outfeed(OpDef):
    """Outfeed operator."""

    op_name = "Outfeed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("AxisIndex")
class AxisIndex(OpDef):
    """AxisIndex operator."""

    op_name = "AxisIndex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("WithShardingConstraint")
class WithShardingConstraint(OpDef):
    """WithShardingConstraint operator."""

    op_name = "WithShardingConstraint"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(args[0], "shape", ()) if args else ()


def infeed(shape: object, dtype: object) -> Tensor:
    """Infeed operation."""
    return _emit_shape_node("Infeed", [], {"shape": shape, "dtype": dtype}, shape, dtype)


def outfeed(tensor: Tensor, token: object = None) -> Tensor:
    """Outfeed operation."""
    return _emit_shape_node("Outfeed", [tensor], {"token": token}, (), tensor.dtype)


def axis_index(axis_name: str) -> Tensor:
    """Axis index operation."""
    from ml_switcheroo_compiler.core.dtype import DType

    return _emit_shape_node("AxisIndex", [], {"axis_name": axis_name}, (), DType.Int32)


def with_sharding_constraint(tensor: Tensor, sharding: object) -> Tensor:
    """With sharding constraint operation."""
    return _emit_shape_node(
        "WithShardingConstraint", [tensor], {"sharding": sharding}, tensor.shape, tensor.dtype
    )
