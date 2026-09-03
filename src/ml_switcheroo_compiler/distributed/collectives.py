"""Zero-copy collective operations for distributed execution over WebRTC."""

import asyncio

import numpy as np


async def _send_data(conn, data: np.ndarray) -> None:
    """Send data over WebRTC DataChannel."""
    # WebRTC RTCDataChannel mock representation for Edge emulation
    shape_str = ",".join(str(s) for s in data.shape)
    dtype_str = str(data.dtype)
    meta = f"{shape_str}|{dtype_str}".encode()
    data_bytes = data.tobytes()

    # Send meta then payload using RTCDataChannel semantics
    await conn.send(meta)
    await conn.send(data_bytes)


async def _recv_data(conn) -> np.ndarray:
    """Receive data over WebRTC DataChannel."""
    meta_bytes = await conn.recv()
    meta_str = meta_bytes.decode("utf-8")
    shape_str, dtype_str = meta_str.split("|")

    shape = tuple(int(s) for s in shape_str.split(",")) if shape_str else ()
    dtype = np.dtype(dtype_str)

    data_bytes = await conn.recv()
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)


async def broadcast(tensor: np.ndarray, root_rank: int, conns: list, rank: int) -> np.ndarray:
    """Execute Broadcast collective operation across WebRTC DataChannels."""
    if rank == root_rank:
        tasks = []
        for i, conn in enumerate(conns):
            if i != rank and conn:
                tasks.append(_send_data(conn, tensor))
        await asyncio.gather(*tasks)
        return tensor.copy()
    else:
        conn = conns[root_rank]
        return await _recv_data(conn)


async def all_reduce(tensor: np.ndarray, op_type: str, conns: list, rank: int) -> np.ndarray:
    """Execute AllReduce collective operation across WebRTC DataChannels."""
    res = tensor.copy()

    # Send to all
    send_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn:
            send_tasks.append(_send_data(conn, tensor))
    await asyncio.gather(*send_tasks)

    # Receive from all
    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn:
            recv_tasks.append(_recv_data(conn))

    received_tensors = await asyncio.gather(*recv_tasks)

    for other_tensor in received_tensors:
        if op_type == "SUM":
            res = res + other_tensor
        elif op_type == "PROD":
            res = res * other_tensor
        elif op_type == "MAX":
            res = np.maximum(res, other_tensor)
        elif op_type == "MIN":
            res = np.minimum(res, other_tensor)

    return res


async def all_gather(tensor: np.ndarray, axis: int, conns: list, rank: int) -> np.ndarray:
    """Execute AllGather collective operation across WebRTC DataChannels."""
    tensors = [None] * len(conns)
    tensors[rank] = tensor.copy()

    send_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn:
            send_tasks.append(_send_data(conn, tensor))
    await asyncio.gather(*send_tasks)

    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn:
            recv_tasks.append(_recv_data(conn))

    received_tensors = await asyncio.gather(*recv_tasks)

    idx = 0
    for i in range(len(conns)):
        if i != rank:
            tensors[i] = received_tensors[idx]
            idx += 1

    valid_tensors = [t for t in tensors if t is not None]
    return np.concatenate(valid_tensors, axis=axis)


async def reduce_scatter(tensor: np.ndarray, op_type: str, scatter_dim: int, conns: list, rank: int) -> np.ndarray:
    """Execute ReduceScatter collective operation across WebRTC DataChannels."""
    reduced = await all_reduce(tensor, op_type, conns, rank)
    world_size = len(conns)
    chunks = np.array_split(reduced, world_size, axis=scatter_dim)
    return chunks[rank]


class DistributedBarrier:
    """A distributed barrier for synchronizing ranks via WebRTC DataChannels."""

    def __init__(self, world_size: int, rank: int, leader_rank: int = 0):
        """Initialize."""
        self.world_size = world_size
        self.rank = rank
        self.leader_rank = leader_rank

    async def wait(self, conns: list):
        """Wait for all ranks to reach the barrier."""
        if self.rank != self.leader_rank:
            conn = conns[self.leader_rank]
            if conn:
                await conn.send(b"r")
                await conn.recv()
        else:
            recv_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn:
                    recv_tasks.append(conn.recv())
            await asyncio.gather(*recv_tasks)

            send_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn:
                    send_tasks.append(conn.send(b"g"))
            await asyncio.gather(*send_tasks)
