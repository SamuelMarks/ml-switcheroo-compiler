"""Distributed collective operations over TCP sockets."""

import pickle
import socket
from typing import Any, Optional

import numpy as np


def _send_data(conn: socket.socket, data: np.ndarray[Any, Any]) -> None:
    """Send data over socket.

    Args:
        conn (socket.socket): The connection.
        data (np.ndarray[Any, Any]): The data.
    """
    payload: object = pickle.dumps(data)
    conn.sendall(len(payload).to_bytes(8, "big") + payload)


def _recv_data(conn: socket.socket) -> np.ndarray[Any, Any]:
    """Receive data from socket.

    Args:
        conn (socket.socket): The connection.

    Returns:
        np.ndarray[Any, Any]: The received data.
    """
    data_len: object = int.from_bytes(conn.recv(8), "big")
    data: object = bytearray()
    while len(data) < data_len:
        chunk: object = conn.recv(min(4096, data_len - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    res: np.ndarray[Any, Any] = pickle.loads(data)
    return res


def all_reduce(tensor: np.ndarray[Any, Any], next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray[Any, Any]:
    """Execute Ring AllReduce collective operation across TCP sockets.

    Args:
        tensor (np.ndarray[Any, Any]): The local tensor.
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray[Any, Any]: The reduced tensor.
    """
    if world_size <= 1:
        return tensor

    chunks: object = np.array_split(tensor, world_size)

    # Scatter-Reduce phase
    send_chunk_idx: object = rank
    recv_chunk_idx: object = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data: object = _recv_data(prev_conn)
        chunks[recv_chunk_idx] += recv_data
        send_chunk_idx: object = (send_chunk_idx - 1) % world_size
        recv_chunk_idx: object = (recv_chunk_idx - 1) % world_size

    # All-Gather phase
    send_chunk_idx: object = (rank + 1) % world_size
    recv_chunk_idx: object = rank

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data: object = _recv_data(prev_conn)
        chunks[recv_chunk_idx] = recv_data
        send_chunk_idx: object = (send_chunk_idx - 1) % world_size
        recv_chunk_idx: object = (recv_chunk_idx - 1) % world_size

    return np.concatenate(chunks)


def all_gather(tensor: np.ndarray[Any, Any], next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray[Any, Any]:
    """Execute Ring AllGather collective operation across TCP sockets.

    Args:
        tensor (np.ndarray[Any, Any]): The local tensor.
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray[Any, Any]: The gathered tensor (concatenated list of arrays).
    """
    if world_size <= 1:
        return tensor

    chunks: list[np.ndarray[Any, Any]] = [np.array([])] * world_size
    chunks[rank] = tensor

    send_chunk_idx: object = rank
    recv_chunk_idx: object = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data: object = _recv_data(prev_conn)
        chunks[recv_chunk_idx] = recv_data
        send_chunk_idx: object = (send_chunk_idx - 1) % world_size
        recv_chunk_idx: object = (recv_chunk_idx - 1) % world_size

    return np.concatenate(chunks)


def reduce_scatter(tensor: np.ndarray[Any, Any], next_conn: socket.socket, prev_conn: socket.socket, rank: int, world_size: int) -> np.ndarray[Any, Any]:
    """Execute Ring ReduceScatter collective operation across TCP sockets.

    Args:
        tensor (np.ndarray[Any, Any]): The local tensor (to be reduced and scattered).
        next_conn (socket.socket): Connection to next rank.
        prev_conn (socket.socket): Connection to previous rank.
        rank (int): Rank of current node.
        world_size (int): Total number of nodes.

    Returns:
        np.ndarray[Any, Any]: The scattered and reduced tensor chunk for this rank.
    """
    if world_size <= 1:
        return np.array_split(tensor, world_size)[0]

    chunks: object = np.array_split(tensor, world_size)

    send_chunk_idx: object = rank
    recv_chunk_idx: object = (rank - 1) % world_size

    for _ in range(world_size - 1):
        _send_data(next_conn, chunks[send_chunk_idx])
        recv_data: object = _recv_data(prev_conn)
        chunks[recv_chunk_idx] += recv_data
        send_chunk_idx: object = (send_chunk_idx - 1) % world_size
        recv_chunk_idx: object = (recv_chunk_idx - 1) % world_size

    return chunks[rank]


def broadcast(tensor: np.ndarray[Any, Any], root_rank: int, conns: list[socket.socket], rank: int) -> np.ndarray[Any, Any]:
    """Execute Broadcast collective operation across TCP sockets.

    Args:
        tensor (np.ndarray[Any, Any]): The local tensor (only relevant if rank == root_rank).
        root_rank (int): The rank of the node broadcasting the tensor.
        conns (list[socket.socket]): List of connections to all other ranks.
        rank (int): Rank of current node.

    Returns:
        np.ndarray[Any, Any]: The broadcasted tensor.
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

        dummy: object = np.array([1], dtype=np.int8)
        for conn in self.conns:
            conn.settimeout(timeout)
            _send_data(conn, dummy)

        for conn in self.conns:
            _recv_data(conn)

        for conn in self.conns:
            conn.settimeout(None)
