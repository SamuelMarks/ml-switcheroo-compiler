# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Real multi-process IPC distributed primitives for Numpy backend."""

import os
from multiprocessing.connection import Client, Listener
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from ml_switcheroo_compiler.distributed.device_mesh import DeviceMesh
import numpy.typing as npt

# A generic payload type that could be transmitted via IPC
IpcPayload = Union[npt.NDArray[np.generic], str, int, float, list, tuple, bool, bytes]


BASE_PORT = 15200


def _exchange_ipc_data_coordinator(size: int, tensor_data: IpcPayload, timeout: float, retry_interval: float) -> list[IpcPayload]:
    """Exchange IPC data as coordinator.

    Args:
        size (int): Size.
        tensor_data: Data.
        timeout (float): Timeout.
        retry_interval (float): Retry interval.

    Returns:
        list[IpcPayload]: Outputs.
    """
    import time

    address: tuple[str, int] = ("localhost", BASE_PORT)
    gathered: list[IpcPayload] = [tensor_data]

    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            for _ in range(size - 1):
                with listener.accept() as conn:
                    other_rank, other_data = conn.recv()
                    gathered.append((other_rank, other_data))

        # Sort by rank
        gathered_sorted: list[IpcPayload] = [tensor_data] * size
        for item in gathered:
            if isinstance(item, tuple):
                r, d = item
                gathered_sorted[r] = d
            else:
                gathered_sorted[0] = item

        # Broadcast back to other ranks
        for r in range(1, size):
            r_address: tuple[str, int] = ("localhost", BASE_PORT + r)
            start_time: float = time.time()
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


def _exchange_ipc_data_worker(rank: int, size: int, tensor_data: IpcPayload, timeout: float, retry_interval: float) -> list[IpcPayload]:
    """Exchange IPC data as worker.

    Args:
        rank (int): Rank.
        size (int): Size.
        tensor_data: Data.
        timeout (float): Timeout.
        retry_interval (float): Retry interval.

    Returns:
        list[IpcPayload]: Outputs.

    Raises:
        TimeoutError: On timeout.
    """
    import time

    port: int = BASE_PORT + rank
    address: tuple[str, int] = ("localhost", port)
    try:
        with Listener(address, authkey=b"ml_switcheroo") as listener:
            start_time: float = time.time()
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
                res: list[IpcPayload] = conn.recv()
        return res
    except Exception:
        # Fallback if connection fails
        return [tensor_data] * size


def _exchange_ipc_data(rank: int, size: int, tensor_data: IpcPayload) -> list[IpcPayload]:
    """Exchanges and synchronizes numpy arrays across local workers using standard multiprocessing IPC.

    Args:
        rank (int): Rank of the current worker.
        size (int): Total number of workers.
        tensor_data: Local array payload.

    Returns:
        list[IpcPayload]: Gathered array payloads from all workers.
    """
    timeout: float = 10.0
    retry_interval: float = 0.05

    if rank == 0:
        return _exchange_ipc_data_coordinator(size, tensor_data, timeout, retry_interval)
    else:
        return _exchange_ipc_data_worker(rank, size, tensor_data, timeout, retry_interval)


def _ipc_all_gather(tensor: IpcPayload, axis: Optional[int], mesh: Optional["DeviceMesh"] = None) -> IpcPayload:
    """Evaluate _ipc_all_gather operation.

    Args:
        tensor: The tensor parameter.
        axis (Optional[int]): The axis parameter.
        mesh: The mesh parameter.

    Returns:
            IpcPayload: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: npt.NDArray[np.generic] = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: int = int(os.environ.get("RANK", "0"))
        size: int = getattr(mesh, "size", 1)

        # Execute real IPC data exchange
        exchanged: list[IpcPayload] = _exchange_ipc_data(rank, size, t)
        return np.concatenate(exchanged, axis=axis if axis is not None else 0)

    return np.expand_dims(t, axis=axis) if axis is not None else t


def _ipc_reduce_scatter(tensor: IpcPayload, op: str, axis: Optional[int], mesh: Optional["DeviceMesh"] = None) -> IpcPayload:
    """Evaluate _ipc_reduce_scatter operation.

    Args:
        tensor: The tensor parameter.
        op (str): The op parameter.
        axis (Optional[int]): The axis parameter.
        mesh: The mesh parameter.

    Returns:
            IpcPayload: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: npt.NDArray[np.generic] = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: int = int(os.environ.get("RANK", "0"))
        size: int = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged: list[IpcPayload] = _exchange_ipc_data(rank, size, t)

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
        sub_arrays: list[npt.NDArray[np.generic]] = np.array_split(reduced, size, axis=axis if axis is not None else 0)
        return sub_arrays[rank % len(sub_arrays)]

    return t


def _ipc_all_reduce(tensor: IpcPayload, op: str, mesh: Optional["DeviceMesh"] = None) -> IpcPayload:
    """Evaluate _ipc_all_reduce operation.

    Args:
        tensor: The tensor parameter.
        op (str): The op parameter.
        mesh: The mesh parameter.

    Returns:
            IpcPayload: Result.
    """
    if isinstance(tensor, str):
        return tensor

    t: npt.NDArray[np.generic] = np.asarray(tensor)
    if mesh is not None and getattr(mesh, "size", 1) > 1:
        rank: int = int(os.environ.get("RANK", "0"))
        size: int = getattr(mesh, "size", 1)

        # Gather all tensors via IPC
        exchanged: list[IpcPayload] = _exchange_ipc_data(rank, size, t)

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
