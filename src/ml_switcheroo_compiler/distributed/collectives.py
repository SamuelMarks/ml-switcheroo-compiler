"""Distributed collective operations over TCP sockets."""

import pickle
import socket
from typing import Optional

import numpy as np


def _send_data(conn: socket.socket, data: np.ndarray) -> None:
    """Send data over socket.

    Args:
        conn (socket.socket): The connection.
        data (np.ndarray): The data.
    """
    payload = pickle.dumps(data)
    conn.sendall(len(payload).to_bytes(8, "big") + payload)


def _recv_data(conn: socket.socket) -> np.ndarray:
    """Receive data from socket.

    Args:
        conn (socket.socket): The connection.

    Returns:
        np.ndarray: The received data.
    """
    data_len = int.from_bytes(conn.recv(8), "big")
    data = bytearray()
    while len(data) < data_len:
        chunk = conn.recv(min(4096, data_len - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    res: np.ndarray = pickle.loads(data)
    return res


def all_reduce(tensor: np.ndarray, next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray:
    """Execute Ring AllReduce collective operation across TCP sockets.

    Args:
        tensor (np.ndarray): The local tensor.
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray: The reduced tensor.
    """
    if world_size <= 1:
        return tensor

    chunks = np.array_split(tensor, world_size)

    # Scatter-Reduce phase
    send_chunk_idx = rank
    recv_chunk_idx = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data = _recv_data(prev_conn)
        chunks[recv_chunk_idx] += recv_data
        send_chunk_idx = (send_chunk_idx - 1) % world_size
        recv_chunk_idx = (recv_chunk_idx - 1) % world_size

    # All-Gather phase
    send_chunk_idx = (rank + 1) % world_size
    recv_chunk_idx = rank

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data = _recv_data(prev_conn)
        chunks[recv_chunk_idx] = recv_data
        send_chunk_idx = (send_chunk_idx - 1) % world_size
        recv_chunk_idx = (recv_chunk_idx - 1) % world_size

    return np.concatenate(chunks)


def all_gather(tensor: np.ndarray, next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray:
    """Execute Ring AllGather collective operation across TCP sockets.

    Args:
        tensor (np.ndarray): The local tensor.
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray: The gathered tensor (concatenated list of arrays).
    """
    if world_size <= 1:
        return tensor

    chunks: list[np.ndarray] = [np.array([])] * world_size
    chunks[rank] = tensor

    send_chunk_idx = rank
    recv_chunk_idx = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data = _recv_data(prev_conn)
        chunks[recv_chunk_idx] = recv_data
        send_chunk_idx = (send_chunk_idx - 1) % world_size
        recv_chunk_idx = (recv_chunk_idx - 1) % world_size

    return np.concatenate(chunks)


def reduce_scatter(tensor: np.ndarray, next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray:
    """Execute Ring ReduceScatter collective operation across TCP sockets.

    Args:
        tensor (np.ndarray): The local tensor (to be reduced and scattered).
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray: The scattered and reduced tensor chunk for this rank.
    """
    if world_size <= 1:
        return np.array_split(tensor, world_size)[0]

    chunks = np.array_split(tensor, world_size)

    send_chunk_idx = rank
    recv_chunk_idx = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data = _recv_data(prev_conn)
        chunks[recv_chunk_idx] += recv_data
        send_chunk_idx = (send_chunk_idx - 1) % world_size
        recv_chunk_idx = (recv_chunk_idx - 1) % world_size

    return chunks[rank]


def broadcast(tensor: np.ndarray, root_rank: int, conns: list[socket.socket], rank: int) -> np.ndarray:
    """Execute Broadcast collective operation across TCP sockets.

    Args:
        tensor (np.ndarray): The local tensor (only relevant if rank == root_rank).
        root_rank (int): The rank of the node broadcasting the tensor.
        conns (list[socket.socket]): List of connections to all other ranks.
        rank (int): Rank of current node.

    Returns:
        np.ndarray: The broadcasted tensor.
    """
    if rank == root_rank:
        for conn in conns:
            _send_data(conn, tensor)
        return tensor
    else:
        # For simplicity, let's say `conns[0]` is the root connection if rank != root_rank.
        return _recv_data(conns[0])


class DistributedBarrier:
    """Connection state management and distributed barriers for training loops."""

    def __init__(self, rank: int, world_size: int, conns: list[socket.socket]):
        """Initialize the barrier."""
        self.rank = rank
        self.world_size = world_size
        self.conns = conns

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until all nodes reach the barrier.

        Args:
            timeout (float): Timeout in seconds.
        """
        if self.world_size <= 1:
            return

        dummy = np.array([1], dtype=np.int8)
        for conn in self.conns:
            conn.settimeout(timeout)
            _send_data(conn, dummy)

        for conn in self.conns:
            _recv_data(conn)

        for conn in self.conns:
            conn.settimeout(None)
