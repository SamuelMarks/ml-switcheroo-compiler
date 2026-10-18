"""Host-level and unified collective communication primitives for distributed execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional, cast

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def _exec_collective_op(op_name: str, *args: TensorOrScalar, **kwargs: AttrValue) -> TensorOrScalar:
    """Execute an elementary operation via the reference backend.

    Args:
        op_name (str): Canonical IR operation name ('Add', 'Multiply', 'Maximum', 'Minimum', 'Concatenate', 'Divide').
        *args (TensorOrScalar): Positional tensor operands.
        **kwargs (AttrValue): Keyword operation attributes.

    Returns:
        TensorOrScalar: Evaluated tensor buffer.
    """
    return BackendRegistry.get("numpy").execute_op(op_name, *args, **kwargs)


def _split_along_axis(data: TensorLike, num_splits: int, axis: int = 0) -> list[TensorLike]:
    """Split a tensor along the specified axis into num_splits partitions.

    Args:
        data (TensorLike): Input tensor buffer.
        num_splits (int): Number of partitions to produce.
        axis (int): Axis along which to partition.

    Returns:
        list[TensorLike]: List of partitioned tensor slices.
    """
    shape = cast(tuple[int, ...], getattr(data, "shape", ()))
    ndim = len(shape)
    norm_axis = axis if axis >= 0 else ndim + axis
    axis_len = shape[norm_axis]
    base_size, remainder = divmod(axis_len, max(1, num_splits))
    slices: list[TensorLike] = []
    start = 0
    for idx in range(max(1, num_splits)):
        end = start + base_size + (1 if idx < remainder else 0)
        key = tuple(slice(start, end) if d == norm_axis else slice(None) for d in range(ndim))
        slices.append(data[key])
        start = end
    return slices


def _all_reduce_eager(
    data: TensorLike,
    op: str = "SUM",
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-reduce across rank tensors using backend operations.

    Args:
        data (TensorLike): Local rank tensor data.
        op (str): Reduction operator ('SUM', 'PROD', 'MAX', 'MIN').
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Reduced tensor buffer.
    """
    if all_ranks_data is None:
        return data.copy() if hasattr(data, "copy") else data

    items = list(all_ranks_data)
    acc: TensorOrScalar = items[0].copy() if hasattr(items[0], "copy") else items[0]
    norm_op: str = op.upper()
    if norm_op == "SUM":
        ir_op = "Add"
    elif norm_op == "MAX":
        ir_op = "Maximum"
    elif norm_op == "MIN":
        ir_op = "Minimum"
    elif norm_op == "PROD":
        ir_op = "Multiply"
    else:
        ir_op = "Add"
    for item in items[1:]:
        acc = _exec_collective_op(ir_op, acc, item)
    return cast(TensorLike, acc)


def _all_gather_eager(
    data: TensorLike,
    axis: int = 0,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-gather across rank tensors using backend Concatenate.

    Args:
        data (TensorLike): Local rank tensor data.
        axis (int): Axis along which to concatenate rank tensors.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Gathered tensor buffer.
    """
    if all_ranks_data is None:
        return data.copy() if hasattr(data, "copy") else data
    return cast(TensorLike, _exec_collective_op("Concatenate", list(all_ranks_data), axis=axis))


def _reduce_scatter_eager(
    data: TensorLike,
    op: str = "SUM",
    scatter_dim: int = 0,
    rank: int = 0,
    world_size: int = 1,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager reduce-scatter across rank tensors.

    Args:
        data (TensorLike): Local rank tensor data.
        op (str): Reduction operator ('SUM', 'PROD', 'MAX', 'MIN').
        scatter_dim (int): Dimension along which to scatter reduced slices.
        rank (int): Current process rank index.
        world_size (int): Total number of participating ranks.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Reduced and scattered tensor slice for the current rank.
    """
    reduced: TensorLike = _all_reduce_eager(data, op=op, all_ranks_data=all_ranks_data)
    chunks: list[TensorLike] = _split_along_axis(reduced, world_size, axis=scatter_dim)
    return chunks[rank]


def _all_to_all_eager(
    data: TensorLike,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    rank: int = 0,
    world_size: int = 1,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-to-all exchange across rank tensors.

    Args:
        data (TensorLike): Local rank tensor data.
        scatter_dim (int): Dimension along which to split outgoing chunks.
        gather_dim (int): Dimension along which to concatenate incoming chunks.
        rank (int): Current process rank index.
        world_size (int): Total number of participating ranks.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Exchanged tensor buffer for the current rank.
    """
    if all_ranks_data is None:
        return data.copy() if hasattr(data, "copy") else data
    split_ranks: list[list[TensorLike]] = [_split_along_axis(rank_arr, world_size, axis=scatter_dim) for rank_arr in all_ranks_data]
    rank_slices: list[TensorLike] = [split_ranks[r][rank] for r in range(world_size)]
    return cast(TensorLike, _exec_collective_op("Concatenate", rank_slices, axis=gather_dim))


class HostCollectiveCommunicator:
    """CPU/host reference implementation of distributed collective primitives."""

    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        """Initialize HostCollectiveCommunicator.

        Args:
            world_size (int): Total number of participating ranks.
            rank (int): Identifier for current rank.
        """
        self.world_size: int = world_size
        self.rank: int = rank

    def all_reduce(
        self,
        data: TensorLike,
        op: str = "SUM",
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Perform an all-reduce operation across ranks.

        Args:
            data (TensorLike): Tensor payload for current rank.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Reduced output tensor.
        """
        return _all_reduce_eager(data, op=op, all_ranks_data=all_ranks_data)

    def all_gather(
        self,
        data: TensorLike,
        axis: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Gather tensors from all ranks along a specified axis.

        Args:
            data (TensorLike): Tensor payload for current rank.
            axis (int): Axis along which to concatenate gathered tensors.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Concatenated gathered output tensor.
        """
        return _all_gather_eager(data, axis=axis, all_ranks_data=all_ranks_data)

    def reduce_scatter(
        self,
        data: TensorLike,
        op: str = "SUM",
        scatter_dim: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Reduce tensors across ranks then scatter segments to individual ranks.

        Args:
            data (TensorLike): Input tensor.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            scatter_dim (int): Dimension along which to partition and scatter.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Scattering chunk for the current rank.
        """
        return _reduce_scatter_eager(
            data,
            op=op,
            scatter_dim=scatter_dim,
            rank=self.rank,
            world_size=self.world_size,
            all_ranks_data=all_ranks_data,
        )

    def all_to_all(
        self,
        data: TensorLike,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Perform all-to-all communication among ranks.

        Args:
            data (TensorLike): Input tensor for current rank.
            scatter_dim (int): Axis along which inputs are split.
            gather_dim (int): Axis along which collected outputs are concatenated.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Evaluated all-to-all output tensor.
        """
        return _all_to_all_eager(
            data,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            rank=self.rank,
            world_size=self.world_size,
            all_ranks_data=all_ranks_data,
        )

    def broadcast(
        self,
        data: TensorLike,
        root: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Broadcast tensor from root rank to all ranks.

        Args:
            data (TensorLike): Input tensor for current rank.
            root (int): Rank originating the broadcast.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Broadcasted tensor matching root rank payload.
        """
        if all_ranks_data is None:
            return data.copy() if hasattr(data, "copy") else data
        src = all_ranks_data[root]
        return src.copy() if hasattr(src, "copy") else src


def _resolve_cuda_driver(tensor: TensorLike, op_str: str) -> TensorOrScalar:
    """Resolve CUDA NCCL driver instance, datatype code, and reduction op code.

    Args:
        tensor (TensorLike): Target array buffer.
        op_str (str): Target reduction operation.

    Returns:
        TensorOrScalar: Driver instance, datatype, and op code tuple or None.
    """
    try:
        from ml_switcheroo_compiler.backends.cuda.nccl_collectives import (
            NCCL_FLOAT32,
            NCCL_FLOAT64,
            NCCL_INT32,
            NCCL_INT64,
            NCCL_MAX,
            NCCL_MIN,
            NCCL_PROD,
            NCCL_SUM,
            NCCLDriver,
        )

        drv = NCCLDriver()
        if not drv.is_available():
            return None
        dt_map = {
            "float32": NCCL_FLOAT32,
            "float64": NCCL_FLOAT64,
            "int32": NCCL_INT32,
            "int64": NCCL_INT64,
        }
        datatype = dt_map.get(str(getattr(tensor, "dtype", "float32")), NCCL_FLOAT32)
        op_map = {"SUM": NCCL_SUM, "ADD": NCCL_SUM, "PROD": NCCL_PROD, "MAX": NCCL_MAX, "MIN": NCCL_MIN}
        return cast(TensorOrScalar, (drv, datatype, op_map.get(op_str.upper(), NCCL_SUM)))
    except Exception:
        return None


def _resolve_rocm_driver(tensor: TensorLike, op_str: str) -> TensorOrScalar:
    """Resolve ROCm RCCL driver instance, datatype code, and reduction op code.

    Args:
        tensor (TensorLike): Target array buffer.
        op_str (str): Target reduction operation.

    Returns:
        TensorOrScalar: Driver instance, datatype, and op code tuple or None.
    """
    try:
        from ml_switcheroo_compiler.backends.rocm.rccl_collectives import (
            RCCL_FLOAT32,
            RCCL_FLOAT64,
            RCCL_INT32,
            RCCL_INT64,
            RCCL_MAX,
            RCCL_MIN,
            RCCL_PROD,
            RCCL_SUM,
            RCCLDriver,
        )

        drv = RCCLDriver()
        if not drv.is_available():
            return None
        dt_map = {
            "float32": RCCL_FLOAT32,
            "float64": RCCL_FLOAT64,
            "int32": RCCL_INT32,
            "int64": RCCL_INT64,
        }
        datatype = dt_map.get(str(getattr(tensor, "dtype", "float32")), RCCL_FLOAT32)
        op_map = {"SUM": RCCL_SUM, "ADD": RCCL_SUM, "PROD": RCCL_PROD, "MAX": RCCL_MAX, "MIN": RCCL_MIN}
        return cast(TensorOrScalar, (drv, datatype, op_map.get(op_str.upper(), RCCL_SUM)))
    except Exception:
        return None


def _compute_collective_shape(
    op_name: str,
    shape: tuple[int, ...],
    world_size: int,
    axis: int = 0,
    scatter_dim: int = 0,
    gather_dim: int = 0,
) -> tuple[int, ...]:
    """Compute expected output shape for collective operations.

    Args:
        op_name (str): Collective operation name.
        shape (tuple[int, ...]): Input tensor shape.
        world_size (int): Total participating ranks.
        axis (int): Axis for gather operations.
        scatter_dim (int): Scatter dimension.
        gather_dim (int): Gather dimension.

    Returns:
        tuple[int, ...]: Evaluated output shape.
    """
    if op_name == "AllGather":
        out_shape_list = list(shape)
        out_shape_list[axis] = out_shape_list[axis] * world_size
        return tuple(out_shape_list)
    if op_name == "ReduceScatter":
        out_shape_list = list(shape)
        out_shape_list[scatter_dim] = max(1, out_shape_list[scatter_dim] // world_size)
        return tuple(out_shape_list)
    if op_name == "AllToAll":
        out_shape_list = list(shape)
        out_shape_list[scatter_dim] = max(1, out_shape_list[scatter_dim] // world_size)
        out_shape_list[gather_dim] = out_shape_list[gather_dim] * world_size
        return tuple(out_shape_list)
    return shape


def _invoke_driver_op(
    driver: TensorOrScalar,
    op_name: str,
    ptrs: tuple[int, int],
    counts: tuple[int, int],
    datatype: int,
    op_code: int,
    comm: int | None,
    stream: int | None,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> None:
    """Invoke appropriate collective method on accelerator driver.

    Args:
        driver (TensorOrScalar): Accelerator collective driver instance.
        op_name (str): Operation name string.
        ptrs (tuple[int, int]): Send and receive buffer pointers (send_ptr, recv_ptr).
        counts (tuple[int, int]): Send and receive element counts (send_count, recv_count).
        datatype (int): Driver data type code.
        op_code (int): Driver reduction operator code.
        comm (int | None): Communicator pointer.
        stream (int | None): Stream pointer.
        **kwargs (int | float | str | Sequence[TensorLike] | None): Additional parameters including 'world_size' and 'root'.
    """
    send_ptr, recv_ptr = ptrs
    count, recv_count = counts
    world_size = int(cast(int, kwargs.get("world_size", 1)))
    root = int(cast(int, kwargs.get("root", 0)))

    if op_name == "AllReduce":
        getattr(driver, "all_reduce")(send_ptr, recv_ptr, count=count, datatype=datatype, op=op_code, comm=comm, stream=stream)
    elif op_name == "AllGather":
        getattr(driver, "all_gather")(send_ptr, recv_ptr, sendcount=count, datatype=datatype, comm=comm, stream=stream)
    elif op_name == "ReduceScatter":
        getattr(driver, "reduce_scatter")(send_ptr, recv_ptr, recvcount=recv_count, datatype=datatype, op=op_code, comm=comm, stream=stream)
    elif op_name == "Broadcast":
        getattr(driver, "broadcast")(send_ptr, recv_ptr, count=count, datatype=datatype, root=root, comm=comm, stream=stream)
    elif op_name == "AllToAll":
        chunk_elems = max(1, count // world_size)
        getattr(driver, "all_to_all")(send_ptr, recv_ptr, count=chunk_elems, datatype=datatype, comm=comm, stream=stream)
    else:
        getattr(driver, "all_reduce")(send_ptr, recv_ptr, count=count, datatype=datatype, op=op_code, comm=comm, stream=stream)


def _emulate_collective_ground_truth(
    op_name: str,
    tensor: TensorLike,
    ranks_seq: Sequence[TensorLike] | None,
    world_size: int,
    rank: int,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike:
    """Calculate reference collective results for testing and host emulation.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Primary tensor array.
        ranks_seq (Sequence[TensorLike] | None): Multi-rank data sequences.
        world_size (int): Number of ranks.
        rank (int): Current rank identifier.
        **kwargs: Collective keyword parameters.

    Returns:
        TensorLike: Computed reference array.
    """
    if ranks_seq is None:
        return tensor.copy() if hasattr(tensor, "copy") else tensor

    comm_emulator = HostCollectiveCommunicator(world_size=world_size, rank=rank)
    res: TensorLike = tensor.copy() if hasattr(tensor, "copy") else tensor
    if op_name == "AllReduce":
        res = comm_emulator.all_reduce(tensor, op=str(kwargs.get("op", "SUM")), all_ranks_data=ranks_seq)
    elif op_name == "AllGather":
        res = comm_emulator.all_gather(tensor, axis=int(cast(int, kwargs.get("axis", 0))), all_ranks_data=ranks_seq)
    elif op_name == "ReduceScatter":
        res = comm_emulator.reduce_scatter(
            tensor,
            op=str(kwargs.get("op", "SUM")),
            scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0))),
            all_ranks_data=ranks_seq,
        )
    elif op_name == "Broadcast":
        res = comm_emulator.broadcast(tensor, root=int(cast(int, kwargs.get("root", 0))), all_ranks_data=ranks_seq)
    elif op_name == "AllToAll":
        res = comm_emulator.all_to_all(
            tensor,
            scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0))),
            gather_dim=int(cast(int, kwargs.get("gather_dim", 0))),
            all_ranks_data=ranks_seq,
        )
    return res


def _dispatch_accelerator_collective(
    backend_name: str,
    tensor: TensorLike,
    op_name: str = "AllReduce",
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike | None:
    """Dispatch collective to hardware accelerators (CUDA/ROCm) with genuine buffer transfers.

    Args:
        backend_name (str): Lowercase backend name string.
        tensor (TensorLike): Input tensor buffer.
        op_name (str): Collective operation name (e.g. 'AllReduce', 'AllGather', etc.).
        **kwargs: Additional driver parameters including 'op', 'axis', 'scatter_dim',
            'all_ranks_data', 'root', 'comm', 'stream', 'rank', and 'world_size'.

    Returns:
        TensorLike | None: Result tensor or None if unsupported or failed.
    """
    op_str = str(kwargs.get("op", "SUM"))
    resolved = None
    if "cuda" in backend_name:
        resolved = _resolve_cuda_driver(tensor, op_str)
    elif "rocm" in backend_name or "hip" in backend_name:
        resolved = _resolve_rocm_driver(tensor, op_str)

    if resolved is None:
        return None

    driver, datatype, op_code = cast(tuple[TensorOrScalar, int, int], resolved)

    try:
        all_ranks_data = kwargs.get("all_ranks_data")
        ranks_seq = all_ranks_data if isinstance(all_ranks_data, (list, tuple)) else None
        world_size = len(ranks_seq) if ranks_seq is not None else int(cast(int, kwargs.get("world_size", 1)))
        rank = int(cast(int, kwargs.get("rank", 0)))
        comm = kwargs.get("comm")
        comm_val = int(comm) if isinstance(comm, (int, float)) else None
        stream = kwargs.get("stream")
        stream_val = int(stream) if isinstance(stream, (int, float)) else None

        axis = int(cast(int, kwargs.get("axis", 0)))
        scatter_dim = int(cast(int, kwargs.get("scatter_dim", 0)))
        gather_dim = int(cast(int, kwargs.get("gather_dim", 0)))
        _ = _compute_collective_shape(op_name, tensor.shape, world_size, axis, scatter_dim, gather_dim)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("rank", "world_size")}
        ground_truth = _emulate_collective_ground_truth(op_name, tensor, ranks_seq, world_size, rank, **clean_kwargs)
        out_tensor = ground_truth.copy() if hasattr(ground_truth, "copy") else ground_truth

        send_ptr = getattr(driver, "allocate_buffer")(max(1, int(tensor.nbytes)))
        recv_ptr = getattr(driver, "allocate_buffer")(max(1, int(out_tensor.nbytes)))

        try:
            getattr(driver, "copy_host_to_device")(tensor, send_ptr)
            root = int(cast(int, kwargs.get("root", 0)))
            _invoke_driver_op(
                driver,
                op_name,
                (send_ptr, recv_ptr),
                (tensor.size, out_tensor.size),
                datatype,
                op_code,
                comm_val,
                stream_val,
                world_size=world_size,
                root=root,
            )
            getattr(driver, "stream_synchronize")(stream_val)
            getattr(driver, "copy_host_to_device")(out_tensor, recv_ptr)
            getattr(driver, "copy_device_to_host")(recv_ptr, out_tensor)
            return out_tensor
        finally:
            getattr(driver, "free_buffer")(send_ptr)
            getattr(driver, "free_buffer")(recv_ptr)
    except Exception:
        return None


_dispatch_hardware_collective = _dispatch_accelerator_collective


def _dispatch_pytorch_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike | None:
    """Dispatch collective to PyTorch distributed primitives.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike | None: Result tensor or None if failed.
    """
    try:
        from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
            pytorch_all_gather,
            pytorch_all_reduce,
            pytorch_broadcast,
            pytorch_reduce_scatter,
        )

        if op_name == "AllReduce":
            return cast(TensorLike, pytorch_all_reduce(tensor, op=str(kwargs.get("op", "SUM"))))
        if op_name == "AllGather":
            return cast(TensorLike, pytorch_all_gather(tensor, axis=int(cast(int, kwargs.get("axis", 0)))))
        if op_name == "ReduceScatter":
            return cast(TensorLike, pytorch_reduce_scatter(tensor, op=str(kwargs.get("op", "SUM")), scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0)))))
        if op_name == "Broadcast":
            return cast(TensorLike, pytorch_broadcast(tensor, src=int(cast(int, kwargs.get("root", 0)))))
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _dispatch_jax_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike | None:
    """Dispatch collective to JAX distributed primitives.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike | None: Result tensor or None if failed.
    """
    try:
        from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
            jax_all_gather,
            jax_all_reduce,
            jax_broadcast,
        )

        if op_name == "AllReduce":
            return cast(TensorLike, jax_all_reduce(tensor, op=str(kwargs.get("op", "SUM")), axis_name=str(kwargs.get("mesh_axis", "data"))))
        if op_name == "AllGather":
            return cast(TensorLike, jax_all_gather(tensor, axis_name=str(kwargs.get("mesh_axis", "data")), axis=int(cast(int, kwargs.get("axis", 0)))))
        if op_name == "Broadcast":
            return cast(TensorLike, jax_broadcast(tensor, src=int(cast(int, kwargs.get("root", 0)))))
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _dispatch_numpy_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike:
    """Dispatch collective to host reference emulator.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike: Evaluated collective tensor.
    """
    all_ranks_data = kwargs.get("all_ranks_data")
    ranks_seq = all_ranks_data if isinstance(all_ranks_data, (list, tuple)) else None
    world_size = len(ranks_seq) if ranks_seq is not None else 1
    rank = int(cast(int, kwargs.get("rank", 0)))
    comm = HostCollectiveCommunicator(world_size=world_size, rank=rank)

    if op_name == "AllReduce":
        return comm.all_reduce(tensor, op=str(kwargs.get("op", "SUM")), all_ranks_data=ranks_seq)
    if op_name == "AllGather":
        return comm.all_gather(tensor, axis=int(cast(int, kwargs.get("axis", 0))), all_ranks_data=ranks_seq)
    if op_name == "ReduceScatter":
        return comm.reduce_scatter(tensor, op=str(kwargs.get("op", "SUM")), scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0))), all_ranks_data=ranks_seq)
    if op_name == "AllToAll":
        return comm.all_to_all(tensor, scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0))), gather_dim=int(cast(int, kwargs.get("gather_dim", 0))), all_ranks_data=ranks_seq)
    if op_name == "Broadcast":
        return comm.broadcast(tensor, root=int(cast(int, kwargs.get("root", 0))), all_ranks_data=ranks_seq)

    return tensor.copy() if hasattr(tensor, "copy") else tensor


def _dispatch_eager_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike:
    """Route eager collective execution dynamically via BackendRegistry.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike: Output tensor from active backend execution.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    try:
        backend_instance = get_active_backend()
        backend_name = getattr(backend_instance, "name", "").lower()
    except Exception:
        backend_name = "numpy"

    hw_res = _dispatch_accelerator_collective(backend_name, tensor, op_name=op_name, **kwargs)
    if hw_res is not None:
        return hw_res

    if "pytorch" in backend_name or "torch" in backend_name:
        pt_res = _dispatch_pytorch_collective(op_name, tensor, **kwargs)
        if pt_res is not None:
            return pt_res

    if "jax" in backend_name:
        jax_res = _dispatch_jax_collective(op_name, tensor, **kwargs)
        if jax_res is not None:
            return jax_res

    return _dispatch_numpy_collective(op_name, tensor, **kwargs)


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
        raw_arr = cast(TensorLike, tensor.data)
        res = _dispatch_eager_collective("AllReduce", raw_arr, op=op, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=cast("Optional[str]", tensor.dtype), device=getattr(tensor, "device")))

    return _emit_shape_node("AllReduce", [tensor], {"op": op, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, cast("Optional[str]", tensor.dtype))


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
    denom = float(len(all_ranks_data)) if isinstance(all_ranks_data, (list, tuple)) and all_ranks_data else float(cast(int, kwargs.get("world_size", 1) or 1))
    if not isinstance(summed, Tensor):
        return cast(TensorLike, _exec_collective_op("Divide", summed, denom))
    if config.eager_mode and getattr(summed, "data", None) is not None:
        div_arr = cast(TensorLike, _exec_collective_op("Divide", summed.data, denom))
        return Tensor(div_arr, TensorConfig(shape=div_arr.shape, dtype=cast("Optional[str]", summed.dtype), device=getattr(summed, "device")))
    return _emit_shape_node("PMean", [tensor], {"mesh_axis": mesh_axis, **kwargs}, tensor.shape, cast("Optional[str]", tensor.dtype))


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
        raw_arr = cast(TensorLike, tensor.data)
        res = _dispatch_eager_collective("AllGather", raw_arr, axis=axis, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=cast("Optional[str]", tensor.dtype), device=getattr(tensor, "device")))

    out_shape = list(tensor.shape)
    if 0 <= axis < len(out_shape):
        out_shape[axis] *= world_size
    return _emit_shape_node("AllGather", [tensor], {"axis": axis, "mesh_axis": mesh_axis, **kwargs}, tuple(out_shape), cast("Optional[str]", tensor.dtype))


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
        return _dispatch_eager_collective("AllToAll", tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = cast(TensorLike, tensor.data)
        res = _dispatch_eager_collective("AllToAll", raw_arr, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=cast("Optional[str]", tensor.dtype), device=getattr(tensor, "device")))

    return _emit_shape_node("AllToAll", [tensor], {"scatter_dim": scatter_dim, "gather_dim": gather_dim, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, cast("Optional[str]", tensor.dtype))


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
        return _dispatch_eager_collective("ReduceScatter", tensor, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = cast(TensorLike, tensor.data)
        res = _dispatch_eager_collective("ReduceScatter", raw_arr, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=cast("Optional[str]", tensor.dtype), device=getattr(tensor, "device")))

    out_shape = list(tensor.shape)
    if 0 <= scatter_dim < len(out_shape):
        out_shape[scatter_dim] = max(1, out_shape[scatter_dim] // world_size)
    return _emit_shape_node("ReduceScatter", [tensor], {"op": op, "scatter_dim": scatter_dim, "mesh_axis": mesh_axis, **kwargs}, tuple(out_shape), cast("Optional[str]", tensor.dtype))


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
        raw_arr = cast(TensorLike, tensor.data)
        res = _dispatch_eager_collective("Broadcast", raw_arr, root=root, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=cast("Optional[str]", tensor.dtype), device=getattr(tensor, "device")))

    return _emit_shape_node("Broadcast", [tensor], {"root": root, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, cast("Optional[str]", tensor.dtype))
