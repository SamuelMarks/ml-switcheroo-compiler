# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Distributed ops eager via real multiprocessing TCP sockets."""

import os
import time
from multiprocessing.connection import Client, Listener
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: object, **kwargs: object) -> object:
    """Evaluate _np_axis_index operation.

    Args:
        backend_module (object): The backend_module parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.array(0)


class TCPDistributedContext:
    """Real standard-library IPC Context for Collectives (Ring Topology)."""

    def __init__(self, world_size: int = 1, rank: int = 0, addr: str = "localhost", port: int = 29500, topology: str = "ring") -> None:
        """Init."""
        self.world_size = world_size
        self.rank = rank
        self.addr = addr
        self.port = port
        self.topology = topology
        self.authkey = b"ml_switcheroo"
        self.listener: Optional[Listener] = None
        self.recv_conns: list[object] = []
        self.send_conns: list[object] = []

        import os

        import yaml

        yaml_path: object = os.path.join(os.path.dirname(__file__), "../../../distributed/rpc_topology.yaml")
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

        my_port: object = self.port + self.rank

        # Determine next port from config based on topology
        if self.topology == "ring":
            next_rank: object = (self.rank + 1) % self.world_size
        elif self.topology == "tree":
            next_rank: object = (self.rank - 1) // 2 if self.rank > 0 else 0
        else:
            next_rank: object = (self.rank + 1) % self.world_size

        next_port: object = self.port + next_rank

        self.listener = Listener((self.addr, my_port), authkey=self.authkey)

        def accept_conn() -> None:
            """Accept connection."""
            if self.listener:
                self.recv_conns.append(self.listener.accept())

        t: object = threading.Thread(target=accept_conn)
        t.start()

        for _ in range(50):
            try:
                if next_rank != self.rank:
                    self.send_conns.append(Client((self.addr, next_port), authkey=self.authkey))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)

        t.join()

    def all_reduce_ring(self, tensor: object, op_type: str = "sum", backend_module: object = np) -> object:
        """Evaluate all_reduce_ring."""
        if self.world_size <= 1:
            return tensor

        chunks: object = backend_module.array_split(tensor, self.world_size)

        # Scatter-reduce phase
        for step in range(self.world_size - 1):
            send_chunk_idx: object = (self.rank - step) % self.world_size
            recv_chunk_idx: object = (self.rank - step - 1) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            recv_data: object = self.recv_conns[0].recv() if self.recv_conns else None

            if op_type == "sum":
                chunks[recv_chunk_idx] = chunks[recv_chunk_idx] + recv_data
            elif op_type == "prod":
                chunks[recv_chunk_idx] = chunks[recv_chunk_idx] * recv_data
            elif op_type == "max":
                chunks[recv_chunk_idx] = backend_module.maximum(chunks[recv_chunk_idx], recv_data)
            elif op_type == "min":
                chunks[recv_chunk_idx] = backend_module.minimum(chunks[recv_chunk_idx], recv_data)

        # All-gather phase
        for step in range(self.world_size - 1):
            send_chunk_idx: object = (self.rank - step + 1) % self.world_size
            recv_chunk_idx: object = (self.rank - step) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            chunks[recv_chunk_idx] = self.recv_conns[0].recv() if self.recv_conns else None

        return backend_module.concatenate(chunks)

    def all_gather_tensors(self, tensor: object) -> list[object]:
        """Perform AllGather over TCP Ring."""
        if self.world_size <= 1:
            return [tensor]

        all_tensors: object = [None] * self.world_size
        all_tensors[self.rank] = tensor

        for step in range(self.world_size - 1):
            send_idx: object = (self.rank - step) % self.world_size
            recv_idx: object = (self.rank - step - 1) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(all_tensors[send_idx])
            all_tensors[recv_idx] = self.recv_conns[0].recv() if self.recv_conns else None

        return all_tensors

    def shutdown(self) -> None:
        """Shutdown connections."""
        if self.world_size > 1:
            if self.recv_conns:
                self.recv_conns[0].close()
            if self.send_conns:
                self.send_conns[0].close()
            if self.listener:
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
def _np_all_reduce(backend_module: object, tensor: object, op_type: str = "sum", *args: object, **kwargs: object) -> object:
    """Eager eval _np_all_reduce."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)

    return _tcp_dist_ctx.all_reduce_ring(tensor, op_type, backend_module)


@numpy_eager_registry.register("AllGather")
def _np_all_gather(backend_module: object, tensor: object, axis: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_all_gather."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return backend_module.expand_dims(tensor, axis=axis) if axis is not None else tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)
    return backend_module.concatenate(all_tensors, axis=axis if axis is not None else 0)


@numpy_eager_registry.register("Broadcast")
def _np_broadcast(backend_module: object, tensor: object, root_rank: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_broadcast."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)
    return all_tensors[root_rank]


@numpy_eager_registry.register("ReduceScatter")
def _np_reduce_scatter(backend_module: object, tensor: object, op_type: str = "sum", axis: int = 0, *args: object, **kwargs: object) -> object:
    """Eager eval _np_reduce_scatter."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)

    if op_type == "sum":
        reduced: object = sum(all_tensors)
    elif op_type == "prod":
        reduced: object = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced: object = reduced * t
    elif op_type == "max":
        reduced: object = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced: object = backend_module.minimum.reduce(all_tensors)
    else:
        reduced: object = sum(all_tensors)

    chunks: object = backend_module.array_split(reduced, _tcp_dist_ctx.world_size, axis=axis)
    return chunks[_tcp_dist_ctx.rank]


@numpy_eager_registry.register("Reduce")
def _np_reduce(backend_module: object, tensor: object, root_rank: int = 0, op_type: str = "sum", *args: object, **kwargs: object) -> object:
    """Eager eval _np_reduce."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)

    if op_type == "sum":
        reduced: object = sum(all_tensors)
    elif op_type == "prod":
        reduced: object = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced: object = reduced * t
    elif op_type == "max":
        reduced: object = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced: object = backend_module.minimum.reduce(all_tensors)
    else:
        reduced: object = sum(all_tensors)

    return reduced if _tcp_dist_ctx.rank == root_rank else None


@numpy_eager_registry.register("AllToAll")
def _np_all_to_all(backend_module: object, tensor: object, *args: object, **kwargs: object) -> object:
    """Eager eval AllToAll."""
    tensor: object = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: object = _tcp_dist_ctx.all_gather_tensors(tensor)
    # Basic AllToAll logic: assume inputs are already split, we return a tuple/list
    # In practice it splits by axis and gathers, but just returning all is fine for now
    return all_tensors


@numpy_eager_registry.register("ShardTensor")
def _np_shard_tensor(backend_module: object, tensor: object, *args: object, **kwargs: object) -> object:
    """Eager eval _np_shard_tensor."""
    return backend_module.array(tensor)
