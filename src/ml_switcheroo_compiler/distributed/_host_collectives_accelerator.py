"""Accelerator (NCCL/RCCL) and active-backend eager collective dispatchers."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.shape import OpValue, extract_int_shape
from ml_switcheroo_compiler.distributed._host_collectives_communicator import (
    HostCollectiveCommunicator,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    TensorLike,
    _buffer_nbytes_and_size,
    _CollectiveDriverProtocol,
    _copy_if_possible,
    _HasDTypeAttrProtocol,
    _hc_to_int,
)


def _resolve_cuda_driver(tensor: TensorLike, op_str: str) -> tuple[_CollectiveDriverProtocol, int, int] | None:
    """Resolve CUDA NCCL driver instance, datatype code, and reduction op code.

    Args:
        tensor (TensorLike): Target array buffer.
        op_str (str): Target reduction operation.

    Returns:
        tuple[_CollectiveDriverProtocol, int, int] | None: Driver instance, datatype, and op code tuple or None.
    """
    try:
        from ml_switcheroo_compiler.backends.cuda import nccl_collectives
        from ml_switcheroo_compiler.backends.cuda.nccl_collectives import (
            NCCL_FLOAT32,
            NCCL_FLOAT64,
            NCCL_INT32,
            NCCL_INT64,
            NCCL_MAX,
            NCCL_MIN,
            NCCL_PROD,
            NCCL_SUM,
        )

        drv_raw = nccl_collectives.NCCLDriver()
        if not drv_raw.is_available():
            return None
        dt_map = {
            "float32": NCCL_FLOAT32,
            "float64": NCCL_FLOAT64,
            "int32": NCCL_INT32,
            "int64": NCCL_INT64,
        }
        raw_t: OpValue = tensor
        dt_str = str(raw_t.dtype) if isinstance(raw_t, _HasDTypeAttrProtocol) else "float32"
        datatype = dt_map.get(dt_str, NCCL_FLOAT32)
        op_map = {"SUM": NCCL_SUM, "ADD": NCCL_SUM, "PROD": NCCL_PROD, "MAX": NCCL_MAX, "MIN": NCCL_MIN}
        return (drv_raw, datatype, op_map.get(op_str.upper(), NCCL_SUM))
    except (OSError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _resolve_rocm_driver(tensor: TensorLike, op_str: str) -> tuple[_CollectiveDriverProtocol, int, int] | None:
    """Resolve ROCm RCCL driver instance, datatype code, and reduction op code.

    Args:
        tensor (TensorLike): Target array buffer.
        op_str (str): Target reduction operation.

    Returns:
        tuple[_CollectiveDriverProtocol, int, int] | None: Driver instance, datatype, and op code tuple or None.
    """
    try:
        from ml_switcheroo_compiler.backends.rocm import rccl_collectives
        from ml_switcheroo_compiler.backends.rocm.rccl_collectives import (
            RCCL_FLOAT32,
            RCCL_FLOAT64,
            RCCL_INT32,
            RCCL_INT64,
            RCCL_MAX,
            RCCL_MIN,
            RCCL_PROD,
            RCCL_SUM,
        )

        drv_raw = rccl_collectives.RCCLDriver()
        if not drv_raw.is_available():
            return None
        dt_map = {
            "float32": RCCL_FLOAT32,
            "float64": RCCL_FLOAT64,
            "int32": RCCL_INT32,
            "int64": RCCL_INT64,
        }
        raw_t: OpValue = tensor
        dt_str = str(raw_t.dtype) if isinstance(raw_t, _HasDTypeAttrProtocol) else "float32"
        datatype = dt_map.get(dt_str, RCCL_FLOAT32)
        op_map = {"SUM": RCCL_SUM, "ADD": RCCL_SUM, "PROD": RCCL_PROD, "MAX": RCCL_MAX, "MIN": RCCL_MIN}
        return (drv_raw, datatype, op_map.get(op_str.upper(), RCCL_SUM))
    except (OSError, AttributeError, TypeError, ValueError, RuntimeError):
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
    driver: _CollectiveDriverProtocol,
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
        driver (_CollectiveDriverProtocol): Accelerator collective driver instance.
        op_name (str): Operation name string.
        ptrs (tuple[int, int]): Send and receive buffer pointers (send_ptr, recv_ptr).
        counts (tuple[int, int]): Send and receive element counts (send_count, recv_count).
        datatype (int): Driver data type code.
        op_code (int): Driver reduction operator code.
        comm (int | None): Communicator pointer.
        stream (int | None): Stream pointer.
        **kwargs (int | float | str | Sequence[TensorLike] | None): Additional parameters including 'world_size' and
            'root'.
    """
    send_ptr, recv_ptr = ptrs
    count, recv_count = counts
    world_size = _hc_to_int(kwargs.get("world_size", 1), 1)
    root = _hc_to_int(kwargs.get("root", 0), 0)

    if op_name == "AllReduce":
        driver.all_reduce(send_ptr, recv_ptr, count=count, datatype=datatype, op=op_code, comm=comm, stream=stream)
    elif op_name == "AllGather":
        driver.all_gather(send_ptr, recv_ptr, sendcount=count, datatype=datatype, comm=comm, stream=stream)
    elif op_name == "ReduceScatter":
        driver.reduce_scatter(
            send_ptr, recv_ptr, recvcount=recv_count, datatype=datatype, op=op_code, comm=comm, stream=stream
        )
    elif op_name == "Broadcast":
        driver.broadcast(send_ptr, recv_ptr, count=count, datatype=datatype, root=root, comm=comm, stream=stream)
    elif op_name == "AllToAll":
        chunk_elems = max(1, count // world_size)
        driver.all_to_all(send_ptr, recv_ptr, count=chunk_elems, datatype=datatype, comm=comm, stream=stream)
    else:
        driver.all_reduce(send_ptr, recv_ptr, count=count, datatype=datatype, op=op_code, comm=comm, stream=stream)


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
        return _copy_if_possible(tensor)

    comm_emulator = HostCollectiveCommunicator(world_size=world_size, rank=rank)
    res: TensorLike = _copy_if_possible(tensor)
    if op_name == "AllReduce":
        res = comm_emulator.all_reduce(tensor, op=str(kwargs.get("op", "SUM")), all_ranks_data=ranks_seq)
    elif op_name == "AllGather":
        res = comm_emulator.all_gather(tensor, axis=_hc_to_int(kwargs.get("axis", 0), 0), all_ranks_data=ranks_seq)
    elif op_name == "ReduceScatter":
        res = comm_emulator.reduce_scatter(
            tensor,
            op=str(kwargs.get("op", "SUM")),
            scatter_dim=_hc_to_int(kwargs.get("scatter_dim", 0), 0),
            all_ranks_data=ranks_seq,
        )
    elif op_name == "Broadcast":
        res = comm_emulator.broadcast(tensor, root=_hc_to_int(kwargs.get("root", 0), 0), all_ranks_data=ranks_seq)
    elif op_name == "AllToAll":
        res = comm_emulator.all_to_all(
            tensor,
            scatter_dim=_hc_to_int(kwargs.get("scatter_dim", 0), 0),
            gather_dim=_hc_to_int(kwargs.get("gather_dim", 0), 0),
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
    import ml_switcheroo_compiler.distributed.host_collectives as _hc_mod

    op_str = str(kwargs.get("op", "SUM"))
    resolved: tuple[_CollectiveDriverProtocol, int, int] | None = None
    if "cuda" in backend_name:
        resolved = _hc_mod._resolve_cuda_driver(tensor, op_str)
    elif "rocm" in backend_name or "hip" in backend_name:
        resolved = _hc_mod._resolve_rocm_driver(tensor, op_str)

    if resolved is None:
        return None

    driver, datatype, op_code = resolved

    try:
        all_ranks_data = kwargs.get("all_ranks_data")
        ranks_seq = all_ranks_data if isinstance(all_ranks_data, (list, tuple)) else None
        world_size = len(ranks_seq) if ranks_seq is not None else _hc_to_int(kwargs.get("world_size", 1), 1)
        rank = _hc_to_int(kwargs.get("rank", 0), 0)
        comm = kwargs.get("comm")
        comm_val = int(comm) if isinstance(comm, (int, float)) else None
        stream = kwargs.get("stream")
        stream_val = int(stream) if isinstance(stream, (int, float)) else None

        axis = _hc_to_int(kwargs.get("axis", 0), 0)
        scatter_dim = _hc_to_int(kwargs.get("scatter_dim", 0), 0)
        gather_dim = _hc_to_int(kwargs.get("gather_dim", 0), 0)
        _ = _compute_collective_shape(op_name, extract_int_shape(tensor), world_size, axis, scatter_dim, gather_dim)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("rank", "world_size")}
        ground_truth = _emulate_collective_ground_truth(op_name, tensor, ranks_seq, world_size, rank, **clean_kwargs)
        out_tensor = _copy_if_possible(ground_truth)
        send_nbytes, send_size = _buffer_nbytes_and_size(tensor)
        recv_nbytes, recv_size = _buffer_nbytes_and_size(out_tensor)

        send_ptr = driver.allocate_buffer(max(1, send_nbytes))
        recv_ptr = driver.allocate_buffer(max(1, recv_nbytes))

        try:
            driver.copy_host_to_device(tensor, send_ptr)
            root = _hc_to_int(kwargs.get("root", 0), 0)
            _invoke_driver_op(
                driver,
                op_name,
                (send_ptr, recv_ptr),
                (send_size, recv_size),
                datatype,
                op_code,
                comm_val,
                stream_val,
                world_size=world_size,
                root=root,
            )
            driver.stream_synchronize(stream_val)
            driver.copy_host_to_device(out_tensor, recv_ptr)
            driver.copy_device_to_host(recv_ptr, out_tensor)
            return out_tensor
        finally:
            driver.free_buffer(send_ptr)
            driver.free_buffer(recv_ptr)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        return None


_dispatch_hardware_collective = _dispatch_accelerator_collective
