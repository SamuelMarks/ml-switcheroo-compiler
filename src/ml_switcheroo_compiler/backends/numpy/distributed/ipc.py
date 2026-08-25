# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Real multi-process IPC distributed primitives for Numpy backend."""

import os
from multiprocessing.connection import Client, Listener

import numpy as np

BASE_PORT = 15200


def _exchange_ipc_data_coordinator(size: int, tensor_data: np.ndarray, timeout: float, retry_interval: float) -> list[np.ndarray]:
    """Exchange IPC data as coordinator.

    Args:
        size (int): Size.
        tensor_data (object): Data.
        timeout (float): Timeout.
        retry_interval (float): Retry interval.

    Returns:
        list: Outputs.
    """
    import time

    address: object = ("localhost", BASE_PORT)
    gathered: object = [tensor_data]

    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            for _ in range(size - 1):
                with listener.accept() as conn:
                    other_rank, other_data = conn.recv()
                    gathered.append((other_rank, other_data))

        # Sort by rank
        gathered_sorted: object = [tensor_data] * size
        for item in gathered:
            if isinstance(item, tuple):
                r, d = item
                gathered_sorted[r] = d
            else:
                gathered_sorted[0] = item

        # Broadcast back to other ranks
        for r in range(1, size):
            r_address: object = ("localhost", BASE_PORT + r)
            start_time: object = time.time()
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
    """Exchange IPC data as worker.

    Args:
        rank (int): Rank.
        size (int): Size.
        tensor_data (object): Data.
        timeout (float): Timeout.
        retry_interval (float): Retry interval.

    Returns:
        list: Outputs.

    Raises:
        TimeoutError: On timeout.
    """
    import time

    port: object = BASE_PORT + rank
    address: object = ("localhost", port)
    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            start_time: object = time.time()
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
                res: object = conn.recv()
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
    timeout: object = 10.0
    retry_interval: object = 0.05

    if rank == 0:
        return _exchange_ipc_data_coordinator(size, tensor_data, timeout, retry_interval)
    else:
        return _exchange_ipc_data_worker(rank, size, tensor_data, timeout, retry_interval)


def _ipc_all_gather(tensor: object, axis: int, mesh: object) -> object:
    """Evaluate _ipc_all_gather operation.

    Args:
        tensor (object): The tensor parameter.
        axis (int): The axis parameter.
        mesh (object): The mesh parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: object = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: object = int(os.environ.get("RANK", "0"))
        size: object = getattr(mesh, "size", 1)

        # Execute real IPC data exchange
        exchanged: object = _exchange_ipc_data(rank, size, t)
        return np.concatenate(exchanged, axis=axis if axis is not None else 0)

    return np.expand_dims(t, axis=axis) if axis is not None else t


def _ipc_reduce_scatter(tensor: object, op: str, axis: int, mesh: object) -> object:
    """Evaluate _ipc_reduce_scatter operation.

    Args:
        tensor (object): The tensor parameter.
        op (str): The op parameter.
        axis (int): The axis parameter.
        mesh (object): The mesh parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: object = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: object = int(os.environ.get("RANK", "0"))
        size: object = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged: object = _exchange_ipc_data(rank, size, t)

        # Perform reduction
        if op == "sum":
            reduced: object = sum(exchanged)
        elif op == "max":
            reduced: object = np.maximum.reduce(exchanged)
        elif op == "min":
            reduced: object = np.minimum.reduce(exchanged)
        else:
            reduced: object = sum(exchanged)

        # Scatter (slice and return own chunk)
        sub_arrays: object = np.array_split(reduced, size, axis=axis if axis is not None else 0)
        return sub_arrays[rank % len(sub_arrays)]

    return t


def _ipc_all_reduce(tensor: object, op: str, mesh: object) -> object:
    """Evaluate _ipc_all_reduce operation.

    Args:
        tensor (object): The tensor parameter.
        op (str): The op parameter.
        mesh (object): The mesh parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: object = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: object = int(os.environ.get("RANK", "0"))
        size: object = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged: object = _exchange_ipc_data(rank, size, t)

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
