"""WebRTC ring reduce-scatter, all-to-all exchange, and distributed barrier."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_operand_list,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import all_reduce
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    WebRTCDataChannelStream,
    _as_tensor_seq,
    _recv_data,
    _send_data,
)
from ml_switcheroo_compiler.backends.registry import get_active_backend


async def ring_reduce_scatter(
    tensor: BackendArg,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[BackendArg | None],
    rank: int,
) -> BackendArg:
    """Execute Ring ReduceScatter collective operation across cyclic peer DataChannels.

    Args:
        tensor (BackendArg): Local input tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        BackendArg: The local slice of the globally reduced tensor.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    world_size: int = len(conns)
    if world_size <= 1:
        return adapt_backend_module(tensor).require_op("copy")()

    op_upper: str = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank: int = (rank + 1) % world_size
    prev_rank: int = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    backend = get_active_backend()
    split_chunks = _as_tensor_seq(backend.execute_op("array_split", tensor, world_size, axis=scatter_dim))
    chunks: list[BackendArg] = [adapt_backend_module(c).require_op("copy")() for c in split_chunks]

    def _apply_op(a: BackendArg, b: BackendArg) -> BackendArg:
        """Apply elementwise reduction between two chunks.

        Args:
            a: Input `a` argument (a).
            b: Input `b` argument (b).

        Returns:
            BackendArg: Computed result.
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

    for step in range(world_size - 1):
        send_idx: int = (rank - 1 - step) % world_size
        recv_idx: int = (rank - 2 - step) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = _apply_op(chunks[recv_idx], received_chunk)

    return chunks[rank]


async def reduce_scatter(
    tensor: BackendArg,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[BackendArg | None],
    rank: int,
    algorithm: str = "direct",
) -> BackendArg:
    """Execute ReduceScatter collective operation across WebRTC DataChannels.

    Args:
        tensor (BackendArg): Local tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): Algorithm choice ("direct", "ring").

    Returns:
        BackendArg: The local slice of the globally reduced tensor.
    """
    if algorithm == "ring":
        return await ring_reduce_scatter(tensor, op_type, scatter_dim, conns, rank)

    reduced = await all_reduce(tensor, op_type, conns, rank)
    world_size = len(conns)
    chunks = coerce_operand_list(get_active_backend().execute_op("array_split", reduced, world_size, axis=scatter_dim))
    return chunks[rank]


async def all_to_all(
    tensor: BackendArg,
    scatter_dim: int,
    gather_dim: int,
    conns: Sequence[BackendArg | None],
    rank: int,
) -> BackendArg:
    """Execute AllToAll collective operation across WebRTC DataChannels.

    Each rank splits its input tensor along scatter_dim into world_size equal parts,
    sends part j to rank j, receives part i from rank i, and concatenates
    the received parts along gather_dim.

    Args:
        tensor (BackendArg): Local input tensor to distribute.
        scatter_dim (int): Axis along which to split local tensor.
        gather_dim (int): Axis along which to concatenate received tensors.
        conns (Sequence[BackendArg | None]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        BackendArg: Assembled tensor after all-to-all exchange.
    """
    world_size = len(conns)
    backend = get_active_backend()
    chunks = coerce_operand_list(backend.execute_op("array_split", tensor, world_size, axis=scatter_dim))

    send_tasks = []
    for j, conn in enumerate(conns):
        if j != rank and conn is not None:
            send_tasks.append(_send_data(conn, chunks[j]))
    await asyncio.gather(*send_tasks)

    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            recv_tasks.append(_recv_data(conn))
    received_chunks = await asyncio.gather(*recv_tasks)

    all_chunks: list[BackendArg | None] = [None] * world_size
    all_chunks[rank] = chunks[rank]
    idx = 0
    for i in range(world_size):
        if i != rank:
            all_chunks[i] = received_chunks[idx]
            idx += 1

    valid_chunks = [c for c in all_chunks if c is not None]
    return backend.execute_op("concatenate", valid_chunks, axis=gather_dim)


class DistributedBarrier:
    """A distributed barrier for synchronizing ranks via WebRTC DataChannels."""

    world_size: int
    rank: int
    leader_rank: int

    def __init__(self, world_size: int, rank: int, leader_rank: int = 0) -> None:
        """Initialize DistributedBarrier.

        Args:
            world_size (int): Total number of participating ranks.
            rank (int): Global rank of current node.
            leader_rank (int): Rank designated as synchronization coordinator.
        """
        self.world_size = world_size
        self.rank = rank
        self.leader_rank = leader_rank

    async def wait(self, conns: Sequence[BackendArg | None]) -> None:
        """Wait for all ranks to reach the barrier.

        Args:
            conns (Sequence[BackendArg | None]): Peer connections indexed by rank.
        """
        if self.rank != self.leader_rank:
            conn = conns[self.leader_rank]
            if conn is not None:
                stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                await stream.send_bytes(b"r")
                await stream.recv_bytes()
        else:
            recv_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn is not None:
                    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                    recv_tasks.append(stream.recv_bytes())
            await asyncio.gather(*recv_tasks)

            send_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn is not None:
                    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                    send_tasks.append(stream.send_bytes(b"g"))
            await asyncio.gather(*send_tasks)
