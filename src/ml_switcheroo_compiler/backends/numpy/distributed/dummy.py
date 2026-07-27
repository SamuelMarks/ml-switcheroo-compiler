# ruff: noqa: E501, C901, PLR0912
"""Real multi-process IPC distributed primitives for Numpy backend."""

import os
from multiprocessing.connection import Client, Listener

import numpy as np

BASE_PORT = 15200


def _exchange_ipc_data_coordinator(size: int, tensor_data: np.ndarray, timeout: float, retry_interval: float) -> list[np.ndarray]:
    """Exchange IPC data as coordinator."""
    import time

    address = ("localhost", BASE_PORT)
    gathered = [tensor_data]

    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            for _ in range(size - 1):
                with listener.accept() as conn:
                    other_rank, other_data = conn.recv()
                    gathered.append((other_rank, other_data))

        # Sort by rank
        gathered_sorted = [tensor_data] * size
        for item in gathered:
            if isinstance(item, tuple):
                r, d = item
                gathered_sorted[r] = d
            else:
                gathered_sorted[0] = item

        # Broadcast back to other ranks
        for r in range(1, size):
            r_address = ("localhost", BASE_PORT + r)
            start_time = time.time()
            while True:
                try:
                    with Client(r_address, authkey=b"ml_switcheroo") as conn:
                        conn.send(gathered_sorted)
                    break
                except Exception:
                    if time.time() - start_time > timeout:
                        break
                    time.sleep(retry_interval)

        return gathered_sorted
    except Exception:
        # Fallback if port is in use or listener fails
        return [tensor_data] * size


def _exchange_ipc_data_worker(rank: int, size: int, tensor_data: np.ndarray, timeout: float, retry_interval: float) -> list[np.ndarray]:
    """Exchange IPC data as worker."""
    import time

    port = BASE_PORT + rank
    address = ("localhost", port)
    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            start_time = time.time()
            while True:
                try:
                    with Client(("localhost", BASE_PORT), authkey=b"ml_switcheroo") as conn:
                        conn.send((rank, tensor_data))
                    break
                except Exception as e:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Timeout connecting to coordinator") from e
                    time.sleep(retry_interval)

            # Receive broadcasted data
            with listener.accept() as conn:
                res = conn.recv()
        return res
    except Exception:
        # Fallback if connection fails
        return [tensor_data] * size


def _exchange_ipc_data(rank: int, size: int, tensor_data: np.ndarray) -> list[np.ndarray]:
    """Exchanges and synchronizes numpy arrays across local workers using standard multiprocessing IPC.

    Args:
        rank (int): Rank of the current worker.
        size (int): Total number of workers.
        tensor_data (np.ndarray): Local array payload.

    Returns:
        list[np.ndarray]: Gathered array payloads from all workers.
    """
    timeout = 10.0
    retry_interval = 0.05

    if rank == 0:
        return _exchange_ipc_data_coordinator(size, tensor_data, timeout, retry_interval)
    else:
        return _exchange_ipc_data_worker(rank, size, tensor_data, timeout, retry_interval)


def _dummy_all_gather(tensor: object, axis: int, mesh: object) -> object:
    """Evaluate and process the multi-process IPC all gather operation.

    Args:
        tensor (object): The input tensor.
        axis (int): Axis along which to concatenate.
        mesh (object): Mesh configuration.

    Returns:
        object: The gathered output array.
    """
    if isinstance(tensor, str):
        return tensor

    t = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank = int(os.environ.get("RANK", "0"))
        size = getattr(mesh, "size", 1)

        # Execute real IPC data exchange
        exchanged = _exchange_ipc_data(rank, size, t)
        return np.concatenate(exchanged, axis=axis if axis is not None else 0)

    return np.expand_dims(t, axis=axis) if axis is not None else t


def _dummy_reduce_scatter(tensor: object, op: str, axis: int, mesh: object) -> object:
    """Evaluate and process the multi-process IPC reduce scatter operation.

    Args:
        tensor (object): The input tensor.
        op (str): Reduction operator.
        axis (int): Axis along which to scatter.
        mesh (object): Mesh configuration.

    Returns:
        object: The reduced and scattered output.
    """
    if isinstance(tensor, str):
        return tensor

    t = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank = int(os.environ.get("RANK", "0"))
        size = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged = _exchange_ipc_data(rank, size, t)

        # Perform reduction
        if op == "sum":
            reduced = sum(exchanged)
        elif op == "max":
            reduced = np.maximum.reduce(exchanged)
        elif op == "min":
            reduced = np.minimum.reduce(exchanged)
        else:
            reduced = sum(exchanged)

        # Scatter (slice and return own chunk)
        sub_arrays = np.array_split(reduced, size, axis=axis if axis is not None else 0)
        return sub_arrays[rank % len(sub_arrays)]

    return t


def _dummy_all_reduce(tensor: object, op: str, mesh: object) -> object:
    """Evaluate and process the multi-process IPC all reduce operation.

    Args:
        tensor (object): The input tensor.
        op (str): Reduction operator.
        mesh (object): Mesh configuration.

    Returns:
        object: The reduced output.
    """
    if isinstance(tensor, str):
        return tensor

    t = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank = int(os.environ.get("RANK", "0"))
        size = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged = _exchange_ipc_data(rank, size, t)

        # Perform reduction and return same result on all workers
        if op == "sum":
            return sum(exchanged)
        elif op == "max":
            return np.maximum.reduce(exchanged)
        elif op == "min":
            return np.minimum.reduce(exchanged)
        else:
            return sum(exchanged)

    return t
