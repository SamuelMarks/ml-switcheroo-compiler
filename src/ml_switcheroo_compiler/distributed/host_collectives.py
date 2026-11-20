"""Host-level and unified collective communication primitives for distributed execution."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    ShapeDim,
    extract_int_shape,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _compute_collective_shape as _compute_collective_shape,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _dispatch_accelerator_collective as _dispatch_accelerator_collective,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _dispatch_hardware_collective as _dispatch_hardware_collective,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _emulate_collective_ground_truth as _emulate_collective_ground_truth,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _invoke_driver_op as _invoke_driver_op,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _resolve_cuda_driver as _resolve_cuda_driver,
)
from ml_switcheroo_compiler.distributed._host_collectives_accelerator import (
    _resolve_rocm_driver as _resolve_rocm_driver,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    HostCollectiveCommunicator as HostCollectiveCommunicator,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _all_gather_eager as _all_gather_eager,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _all_reduce_eager as _all_reduce_eager,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _all_to_all_eager as _all_to_all_eager,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _as_sliceable_buffer as _as_sliceable_buffer,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _exec_collective_op as _exec_collective_op,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _reduce_scatter_eager as _reduce_scatter_eager,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _SliceableBufferProtocol as _SliceableBufferProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    _split_along_axis as _split_along_axis,
)
from ml_switcheroo_compiler.distributed._host_collectives_dispatch import (
    _dispatch_eager_collective as _dispatch_eager_collective,
)
from ml_switcheroo_compiler.distributed._host_collectives_dispatch import (
    _dispatch_host_collective as _dispatch_host_collective,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    TensorLike as TensorLike,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _buffer_nbytes_and_size as _buffer_nbytes_and_size,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _CollectiveDriverProtocol as _CollectiveDriverProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _copy_if_possible as _copy_if_possible,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _CopyableProtocol as _CopyableProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _HasDTypeAttrProtocol as _HasDTypeAttrProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _hc_extract_shape as _hc_extract_shape,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _hc_opt_str as _hc_opt_str,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _hc_to_int as _hc_to_int,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _NamedBackendObjProtocol as _NamedBackendObjProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    _SizedDeviceBufferProtocol as _SizedDeviceBufferProtocol,
)
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def all_reduce(
    tensor: Tensor | TensorLike,
    op: str = "SUM",
    mesh_axis: str = "data",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal AllReduce collective operation.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
        mesh_axis (str): Mesh axis name for collective reduction.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Reduced tensor result.
    """
    if not isinstance(tensor, Tensor):
        return _dispatch_eager_collective("AllReduce", tensor, op=op, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        res = _dispatch_eager_collective("AllReduce", tensor.data, op=op, mesh_axis=mesh_axis, **kwargs)
        return Tensor(
            res, TensorConfig(shape=extract_shape(res), dtype=_hc_opt_str(tensor.dtype), device=tensor.device)
        )

    return _emit_shape_node(
        "AllReduce", [tensor], {"op": op, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, _hc_opt_str(tensor.dtype)
    )


def psum(
    tensor: Tensor | TensorLike,
    mesh_axis: str = "data",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal parallel sum (PSum) collective operation across a mesh axis.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        mesh_axis (str): Mesh axis name for collective reduction.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Summed tensor result.
    """
    return all_reduce(tensor, op="SUM", mesh_axis=mesh_axis, **kwargs)


def pmean(
    tensor: Tensor | TensorLike,
    mesh_axis: str = "data",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal parallel mean (PMean) collective operation across a mesh axis.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        mesh_axis (str): Mesh axis name for collective reduction.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Averaged tensor result.
    """
    summed = all_reduce(tensor, op="SUM", mesh_axis=mesh_axis, **kwargs)
    all_ranks_data = kwargs.get("all_ranks_data")
    denom = (
        float(len(all_ranks_data))
        if isinstance(all_ranks_data, (list, tuple)) and all_ranks_data
        else float(_hc_to_int(kwargs.get("world_size", 1) or 1, 1))
    )
    if not isinstance(summed, Tensor):
        return _exec_collective_op("Divide", summed, denom)
    if config.eager_mode and summed.data is not None:
        div_arr = _exec_collective_op("Divide", summed.data, denom)
        return Tensor(
            div_arr,
            TensorConfig(shape=extract_int_shape(div_arr), dtype=_hc_opt_str(summed.dtype), device=summed.device),
        )
    assert isinstance(tensor, Tensor)
    return _emit_shape_node(
        "PMean", [tensor], {"mesh_axis": mesh_axis, **kwargs}, tensor.shape, _hc_opt_str(tensor.dtype)
    )


def all_gather(
    tensor: Tensor | TensorLike,
    axis: int = 0,
    mesh_axis: str = "data",
    world_size: int = 1,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal AllGather collective operation.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        axis (int): Axis along which gathered tensors are concatenated.
        mesh_axis (str): Mesh axis name for collective gathering.
        world_size (int): Number of participating devices.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Concatenated gathered tensor result.
    """
    if not isinstance(tensor, Tensor):
        return _dispatch_eager_collective("AllGather", tensor, axis=axis, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        res = _dispatch_eager_collective("AllGather", tensor.data, axis=axis, mesh_axis=mesh_axis, **kwargs)
        return Tensor(
            res, TensorConfig(shape=extract_shape(res), dtype=_hc_opt_str(tensor.dtype), device=tensor.device)
        )

    out_shape = list(tensor.shape)
    if 0 <= axis < len(out_shape):
        out_shape[axis] *= world_size
    return _emit_shape_node(
        "AllGather",
        [tensor],
        {"axis": axis, "mesh_axis": mesh_axis, **kwargs},
        tuple(out_shape),
        _hc_opt_str(tensor.dtype),
    )


def all_to_all(
    tensor: Tensor | TensorLike,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    mesh_axis: str = "data",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal AllToAll collective operation.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        scatter_dim (int): Axis along which inputs are split.
        gather_dim (int): Axis along which gathered chunks are concatenated.
        mesh_axis (str): Mesh axis name for collective redistribution.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: AllToAll redistributed tensor result.
    """
    if not isinstance(tensor, Tensor):
        return _dispatch_eager_collective(
            "AllToAll", tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs
        )

    if config.eager_mode:
        res = _dispatch_eager_collective(
            "AllToAll", tensor.data, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs
        )
        return Tensor(
            res, TensorConfig(shape=extract_shape(res), dtype=_hc_opt_str(tensor.dtype), device=tensor.device)
        )

    return _emit_shape_node(
        "AllToAll",
        [tensor],
        {"scatter_dim": scatter_dim, "gather_dim": gather_dim, "mesh_axis": mesh_axis, **kwargs},
        tensor.shape,
        _hc_opt_str(tensor.dtype),
    )


def reduce_scatter(
    tensor: Tensor | TensorLike,
    op: str = "SUM",
    scatter_dim: int = 0,
    mesh_axis: str = "data",
    world_size: int = 1,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal ReduceScatter collective operation.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
        scatter_dim (int): Dimension along which reduced tensor is scattered.
        mesh_axis (str): Mesh axis name.
        world_size (int): Number of participating devices.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Scattered reduced chunk tensor.
    """
    if not isinstance(tensor, Tensor):
        return _dispatch_eager_collective(
            "ReduceScatter", tensor, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs
        )

    if config.eager_mode:
        res = _dispatch_eager_collective(
            "ReduceScatter", tensor.data, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs
        )
        return Tensor(
            res, TensorConfig(shape=extract_shape(res), dtype=_hc_opt_str(tensor.dtype), device=tensor.device)
        )

    out_shape: list[ShapeDim] = list(tensor.shape)
    if 0 <= scatter_dim < len(out_shape):
        cur_dim = out_shape[scatter_dim]
        if isinstance(cur_dim, int):
            out_shape[scatter_dim] = max(1, cur_dim // world_size)
    return _emit_shape_node(
        "ReduceScatter",
        [tensor],
        {"op": op, "scatter_dim": scatter_dim, "mesh_axis": mesh_axis, **kwargs},
        tuple(out_shape),
        _hc_opt_str(tensor.dtype),
    )


def broadcast(
    tensor: Tensor | TensorLike,
    root: int = 0,
    mesh_axis: str = "data",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> Tensor | TensorLike:
    """Universal Broadcast collective operation.

    Args:
        tensor (Tensor | TensorLike): Input tensor or array.
        root (int): Root rank originating the broadcast.
        mesh_axis (str): Mesh axis name.
        **kwargs: Additional collective arguments.

    Returns:
        Tensor | TensorLike: Broadcasted tensor matching root payload.
    """
    if not isinstance(tensor, Tensor):
        return _dispatch_eager_collective("Broadcast", tensor, root=root, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        res = _dispatch_eager_collective("Broadcast", tensor.data, root=root, mesh_axis=mesh_axis, **kwargs)
        return Tensor(
            res, TensorConfig(shape=extract_shape(res), dtype=_hc_opt_str(tensor.dtype), device=tensor.device)
        )

    return _emit_shape_node(
        "Broadcast", [tensor], {"root": root, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, _hc_opt_str(tensor.dtype)
    )
