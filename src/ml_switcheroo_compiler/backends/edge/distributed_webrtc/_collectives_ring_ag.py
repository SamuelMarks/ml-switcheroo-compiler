"""WebRTC broadcast, ring all-reduce, and recursive halving-doubling all-gather."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_numeric,
    coerce_operand_list,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _as_tensor_seq,
    _recv_data,
    _send_data,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import get_active_backend


async def broadcast(
    tensor: BackendArg,
    root_rank: int,
    conns: Sequence[BackendArg | None],
    rank: int,
) -> BackendArg:
    """Execute Broadcast collective operation across WebRTC DataChannels.

    Args:
        tensor (BackendArg): Tensor to broadcast from root or receive buffer.
        root_rank (int): Rank of broadcasting leader node.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        BackendArg: Received or copied tensor data.

    Raises:
        ValueError: If root connection is missing.
    """
    if rank == root_rank:
        tasks = []
        for i, conn in enumerate(conns):
            if i != rank and conn is not None:
                tasks.append(_send_data(conn, tensor))
        await asyncio.gather(*tasks)
        return adapt_backend_module(tensor).require_op("copy")()
    else:
        conn = conns[root_rank]
        if conn is None:
            raise ValueError(f"Connection to root rank {root_rank} is not established.")
        return await _recv_data(conn)


async def ring_all_reduce(
    tensor: BackendArg,
    op_type: str,
    conns: Sequence[BackendArg | None],
    rank: int,
) -> BackendArg:
    """Execute Ring AllReduce collective operation across cyclic peer DataChannels.

    Args:
        tensor (BackendArg): Local input tensor to reduce across all ranks.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        BackendArg: Globally reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    world_size = len(conns)
    if world_size <= 1:
        return adapt_backend_module(tensor).require_op("copy")()

    op_upper = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    backend = get_active_backend()
    split_chunks = _as_tensor_seq(backend.execute_op("array_split", tensor, world_size, axis=0))
    chunks: list[BackendArg] = [adapt_backend_module(c).require_op("copy")() for c in split_chunks]

    def _apply_op(a: BackendArg, b: BackendArg) -> BackendArg:
        """Apply elementwise reduction operation between two chunks.

        Args:
            a (BackendArg): First operand chunk.
            b (BackendArg): Second operand chunk.

        Returns:
            BackendArg: Result of the reduction.
        """
        a_t = coerce_numeric(a)
        b_t = coerce_numeric(b)
        if op_upper == "SUM":
            return a_t + b_t
        elif op_upper == "PROD":
            return a_t * b_t
        elif op_upper == "MAX":
            return backend.execute_op("maximum", a_t, b_t)
        else:
            return backend.execute_op("minimum", a_t, b_t)

    # Phase 1: Scatter-Reduce
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = _apply_op(chunks[recv_idx], received_chunk)

    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = received_chunk

    return backend.execute_op("concatenate", chunks, axis=0)


async def recursive_halving_doubling_all_gather(
    tensor: BackendArg,
    axis: int,
    conns: Sequence[BackendArg | None],
    rank: int,
) -> BackendArg:
    """Execute Recursive Halving/Doubling AllGather collective across WebRTC DataChannels.

    Args:
        tensor (BackendArg): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        BackendArg: Concatenated global tensor containing all slices in rank order.
    """
    world_size = len(conns)
    if world_size <= 1:
        return adapt_backend_module(tensor).require_op("copy")()

    backend = get_active_backend()
    gathered: dict[int, BackendArg] = {rank: adapt_backend_module(tensor).require_op("copy")()}
    num_steps = (world_size - 1).bit_length()

    for step in range(num_steps):
        distance = 1 << step
        partner_rank = rank ^ distance
        if partner_rank < world_size:
            partner_conn = conns[partner_rank]
            if partner_conn is not None:
                current_ranks = sorted(gathered.keys())
                current_payload = backend.execute_op("concatenate", [gathered[r] for r in current_ranks], axis=axis)

                send_task = _send_data(partner_conn, current_payload)
                recv_task = _recv_data(partner_conn)
                _, partner_payload = await asyncio.gather(send_task, recv_task)

                partner_ranks = [r ^ distance for r in current_ranks if (r ^ distance) < world_size]
                received_slices = coerce_operand_list(
                    backend.execute_op("array_split", partner_payload, len(partner_ranks), axis=axis)
                )
                for pr, slc in zip(partner_ranks, received_slices):
                    gathered[pr] = slc

    valid_slices = [gathered[r] for r in range(world_size) if r in gathered]
    return backend.execute_op("concatenate", valid_slices, axis=axis)


def _apply_reduction_op(a: BackendArg, b: BackendArg, op_type: str) -> BackendArg:
    """Apply an elementwise reduction operator between two tensors.

    Args:
        a (BackendArg): First operand tensor.
        b (BackendArg): Second operand tensor.
        op_type (str): Reduction operator name ("SUM", "PROD", "MAX", "MIN").

    Returns:
        BackendArg: Reduced result tensor.

    Raises:
        ValueError: If op_type is not supported.
    """
    a_t = coerce_numeric(a)
    b_t = coerce_numeric(b)
    if op_type == "SUM":
        return a_t + b_t
    if op_type == "PROD":
        return a_t * b_t
    if op_type == "MAX":
        return get_active_backend().execute_op("maximum", a_t, b_t)
    if op_type == "MIN":
        return get_active_backend().execute_op("minimum", a_t, b_t)
    raise ValueError(f"Unsupported reduction operator '{op_type}'")


async def all_reduce(
    tensor: BackendArg,
    op_type: str,
    conns: Sequence[BackendArg | None],
    rank: int,
    algorithm: str = "direct",
) -> BackendArg:
    """Execute AllReduce collective operation across WebRTC DataChannels.

    Args:
        tensor (BackendArg): Local input tensor to reduce.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllReduce algorithm choice ("direct", "ring").

    Returns:
        BackendArg: Fully reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized.
    """
    if algorithm == "ring":
        return await ring_all_reduce(tensor, op_type, conns, rank)

    res: BackendArg = adapt_backend_module(tensor).require_op("copy")()
    peer_conns = [conn for i, conn in enumerate(conns) if i != rank and conn is not None]
    await asyncio.gather(*[_send_data(conn, tensor) for conn in peer_conns])
    received_tensors = await asyncio.gather(*[_recv_data(conn) for conn in peer_conns])

    for other_tensor in received_tensors:
        res = _apply_reduction_op(res, other_tensor, op_type)

    return res


async def all_gather(
    tensor: BackendArg,
    axis: int,
    conns: Sequence[BackendArg | None],
    rank: int,
    algorithm: str = "direct",
) -> BackendArg:
    """Execute AllGather collective operation across WebRTC DataChannels.

    Args:
        tensor (BackendArg): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllGather algorithm choice ("direct", "recursive_doubling").

    Returns:
        BackendArg: Concatenated global tensor containing all slices in rank order.
    """
    if algorithm == "recursive_doubling":
        return await recursive_halving_doubling_all_gather(tensor, axis, conns, rank)

    world_size = len(conns)
    tensors: list[BackendArg | None] = [None] * world_size
    tensors[rank] = adapt_backend_module(tensor).require_op("copy")()

    send_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            send_tasks.append(_send_data(conn, tensor))
    await asyncio.gather(*send_tasks)

    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            recv_tasks.append(_recv_data(conn))

    received_tensors = await asyncio.gather(*recv_tasks)

    idx = 0
    for i in range(world_size):
        if i != rank:
            tensors[i] = received_tensors[idx]
            idx += 1

    valid_tensors = [t for t in tensors if t is not None]
    return get_active_backend().execute_op("concatenate", valid_tensors, axis=axis)
