"""Distributed ops eager via real multiprocessing TCP sockets."""

from __future__ import annotations

import os  # noqa: F401
import time
import typing
from multiprocessing.connection import Client, Connection, Listener

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: BackendModuleProtocol, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np axis index operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.array(0)


class TCPDistributedContext:
    """Real standard-library IPC Context for Collectives (Ring Topology)."""

    def __init__(self, world_size: int = 1, rank: int = 0, addr: str = "localhost", port: int = 29500, topology: str = "ring") -> None:
        """Init.

        Args:
            world_size: The world_size parameter.
            rank: The rank parameter.
            addr: The addr parameter.
            port: The port parameter.
            topology: The topology parameter.
        """
        self.world_size = world_size
        self.rank = rank
        self.addr = addr
        self.port = port
        self.topology = topology
        self.authkey = b"ml_switcheroo"
        self.listener: Listener | None = None
        self.recv_conns: list[Connection[EagerValue, EagerValue]] = []
        self.send_conns: list[Connection[EagerValue, EagerValue]] = []

        import os  # noqa: F811

        import yaml

        yaml_path: str = os.path.join(os.path.dirname(__file__), "../../../distributed/rpc_topology.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.config = yaml.safe_load(f).get("topologies", {}).get(topology, {})
        else:
            self.config = {}

    def initialize(self) -> None:
        """Initialize the collective communication ring topology."""
        if self.world_size <= 1:
            return

        import threading

        my_port: int = self.port + self.rank

        # Determine next port from config based on topology
        if self.topology == "ring":
            next_rank: int = (self.rank + 1) % self.world_size
        elif self.topology == "tree":
            next_rank = (self.rank - 1) // 2 if self.rank > 0 else 0
        else:
            next_rank = (self.rank + 1) % self.world_size

        next_port: int = self.port + next_rank

        self.listener = Listener((self.addr, my_port), authkey=self.authkey)

        def accept_conn() -> None:
            """Accept connection."""
            if self.listener:
                self.recv_conns.append(self.listener.accept())

        t: threading.Thread = threading.Thread(target=accept_conn)
        t.start()

        for _ in range(50):
            try:
                if next_rank != self.rank:
                    self.send_conns.append(Client((self.addr, next_port), authkey=self.authkey))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)

        t.join()

    def all_reduce_ring(self, tensor: np.ndarray[tuple[int, ...], np.dtype[np.generic]], op_type: str = "sum", backend_module: BackendModuleProtocol = np) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Evaluate all_reduce_ring.

        Args:
            tensor: The tensor parameter.
            op_type: The op_type parameter.
            backend_module: The backend_module parameter.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Computed result.
        """
        if self.world_size <= 1:
            return tensor

        chunks: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(backend_module, "array_split")(tensor, self.world_size)

        # Scatter-reduce phase
        for step in range(self.world_size - 1):
            send_chunk_idx: int = (self.rank - step) % self.world_size
            recv_chunk_idx: int = (self.rank - step - 1) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            recv_data = typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", self.recv_conns[0].recv()) if self.recv_conns else None

            if op_type == "sum":
                chunks[recv_chunk_idx] = chunks[recv_chunk_idx] + recv_data
            elif op_type == "prod":
                chunks[recv_chunk_idx] = chunks[recv_chunk_idx] * recv_data
            elif op_type == "max":
                chunks[recv_chunk_idx] = getattr(backend_module, "maximum")(chunks[recv_chunk_idx], recv_data)
            elif op_type == "min":
                chunks[recv_chunk_idx] = getattr(backend_module, "minimum")(chunks[recv_chunk_idx], recv_data)

        # All-gather phase
        for step in range(self.world_size - 1):
            send_chunk_idx = (self.rank - step + 1) % self.world_size
            recv_chunk_idx = (self.rank - step) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            chunks[recv_chunk_idx] = self.recv_conns[0].recv() if self.recv_conns else None

        return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(backend_module, "concatenate")(chunks))

    def all_gather_tensors(self, tensor: EagerValue) -> list[EagerValue]:
        """Perform AllGather over TCP Ring.

        Args:
            tensor: The tensor parameter.

        Returns:
            list[EagerValue]: Computed result.
        """
        if self.world_size <= 1:
            return [tensor]

        all_tensors: list[EagerValue] = [None] * self.world_size
        all_tensors[self.rank] = tensor

        for step in range(self.world_size - 1):
            send_idx: int = (self.rank - step) % self.world_size
            recv_idx: int = (self.rank - step - 1) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(all_tensors[send_idx])
            all_tensors[recv_idx] = self.recv_conns[0].recv() if self.recv_conns else None

        return all_tensors

    def shutdown(self) -> None:
        """Shutdown connections."""
        if self.world_size > 1:
            if self.recv_conns and hasattr(self.recv_conns[0], "close"):
                self.recv_conns[0].close()
            if self.send_conns and hasattr(self.send_conns[0], "close"):
                self.send_conns[0].close()
            if self.listener and hasattr(self.listener, "close"):
                self.listener.close()


_tcp_dist_ctx = TCPDistributedContext()


def set_np_distributed_context(world_size: int, rank: int, addr: str = "localhost", port: int = 29500) -> None:
    """Set mock distributed context.

    Args:
        world_size (int): The world_size parameter.
        rank (int): The rank parameter.
        addr (str): The addr parameter.
        port (int): The port parameter.
    """
    global _tcp_dist_ctx
    _tcp_dist_ctx.shutdown()
    _tcp_dist_ctx = TCPDistributedContext(world_size, rank, addr, port)
    _tcp_dist_ctx.initialize()


@numpy_eager_registry.register("AllReduce")
@numpy_eager_registry.register("NcclAllReduce")
@numpy_eager_registry.register("HierarchicalCopyAllReduce")
def _np_all_reduce(backend_module: BackendModuleProtocol, tensor: EagerValue, op_type: str = "sum", *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_all_reduce.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        op_type: The op_type parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    return typing.cast("EagerValue", getattr(_tcp_dist_ctx, "all_reduce_ring")(tensor, op_type, backend_module))


@numpy_eager_registry.register("AllGather")
def _np_all_gather(backend_module: BackendModuleProtocol, tensor: EagerValue, axis: int | None = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_all_gather.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        axis: The axis parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return backend_module.expand_dims(tensor, axis=axis) if axis is not None else tensor

    all_tensors: list[EagerValue] = _tcp_dist_ctx.all_gather_tensors(tensor)
    return backend_module.concatenate(all_tensors, axis=axis if axis is not None else 0)


@numpy_eager_registry.register("Broadcast")
def _np_broadcast(backend_module: BackendModuleProtocol, tensor: EagerValue, root_rank: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_broadcast.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        root_rank: The root_rank parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerValue] = _tcp_dist_ctx.all_gather_tensors(tensor)
    return all_tensors[root_rank]


@numpy_eager_registry.register("ReduceScatter")
def _np_reduce_scatter(backend_module: BackendModuleProtocol, tensor: EagerValue, op_type: str = "sum", axis: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_reduce_scatter.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        op_type: The op_type parameter.
        axis: The axis parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerTensorProtocol] = typing.cast("list[EagerTensorProtocol]", _tcp_dist_ctx.all_gather_tensors(tensor))

    if op_type == "sum":
        reduced: EagerTensorProtocol = backend_module.sum(all_tensors, axis=0)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced = backend_module.minimum.reduce(all_tensors)
    else:
        reduced = backend_module.sum(all_tensors, axis=0)

    chunks: EagerTensorProtocol = backend_module.array_split(reduced, _tcp_dist_ctx.world_size, axis=axis)
    return typing.cast("EagerValue", chunks[_tcp_dist_ctx.rank])


@numpy_eager_registry.register("Reduce")
def _np_reduce(backend_module: BackendModuleProtocol, tensor: EagerValue, root_rank: int = 0, op_type: str = "sum", *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_reduce.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        root_rank: The root_rank parameter.
        op_type: The op_type parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerTensorProtocol] = typing.cast("list[EagerTensorProtocol]", _tcp_dist_ctx.all_gather_tensors(tensor))

    if op_type == "sum":
        reduced: EagerTensorProtocol = backend_module.sum(all_tensors, axis=0)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced = backend_module.minimum.reduce(all_tensors)
    else:
        reduced = backend_module.sum(all_tensors, axis=0)

    return reduced if _tcp_dist_ctx.rank == root_rank else None


@numpy_eager_registry.register("AllToAll")
def _np_all_to_all(backend_module: BackendModuleProtocol, tensor: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval AllToAll.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerValue] = _tcp_dist_ctx.all_gather_tensors(tensor)
    # Basic AllToAll logic: assume inputs are already split, we return a tuple/list
    # In practice it splits by axis and gathers, but just returning all is fine for now
    return all_tensors


@numpy_eager_registry.register("ShardTensor")
def _np_shard_tensor(backend_module: BackendModuleProtocol, tensor: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager eval _np_shard_tensor.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        *args: The args parameter.
        **kwargs: The kwargs parameter.

    Returns:
        EagerValue: Computed result.
    """
    return backend_module.array(tensor)
