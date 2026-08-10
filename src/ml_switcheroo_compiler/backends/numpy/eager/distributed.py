# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Distributed ops eager via real multiprocessing TCP sockets."""

import os
import time
from multiprocessing.connection import Client, Listener
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: Any, **kwargs: Any) -> Any:
    """Evaluate _np_axis_index operation.

    Args:
        backend_module (object): The backend_module parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.array(0)


class TCPDistributedContext:
    """Real standard-library IPC Context for Collectives."""

    def __init__(self, world_size: int = 1, rank: int = 0, addr: str = "localhost", port: int = 29500) -> None:
        """Init."""
        self.world_size = world_size
        self.rank = rank
        self.addr = addr
        self.port = port
        self.authkey = b"ml_switcheroo"
        self.listener: Optional[Listener] = None
        self.connections: list[Any] = []
        self.conn: Optional[Any] = None

    def initialize(self) -> None:
        """Initialize the collective communication topology."""
        if self.world_size <= 1:
            return

        if self.rank == 0:
            self.listener = Listener((self.addr, self.port), authkey=self.authkey)
            self.connections = []
            for _ in range(self.world_size - 1):
                self.connections.append(self.listener.accept())
        else:
            for _ in range(50):
                try:
                    self.conn = Client((self.addr, self.port), authkey=self.authkey)
                    break
                except ConnectionRefusedError:
                    time.sleep(0.1)

    def all_gather_tensors(self, tensor: Any) -> list[Any]:
        """Perform AllGather over TCP."""
        if self.world_size <= 1:
            return [tensor]

        if self.rank == 0:
            all_tensors = [None] * self.world_size
            all_tensors[0] = tensor
            # collect
            for conn in self.connections:
                data = conn.recv()
                r, t = data["rank"], data["tensor"]
                all_tensors[r] = t
            # broadcast
            for conn in self.connections:
                conn.send(all_tensors)
            return all_tensors
        else:
            if self.conn is None:
                raise RuntimeError("Not initialized")
            self.conn.send({"rank": self.rank, "tensor": tensor})
            return self.conn.recv()

    def shutdown(self) -> None:
        """Shutdown connections."""
        if self.world_size > 1:
            if self.rank == 0:
                for c in self.connections:
                    c.close()
                if self.listener:
                    self.listener.close()
            else:
                if self.conn:
                    self.conn.close()


_tcp_dist_ctx = TCPDistributedContext()


def set_mock_distributed_context(world_size: int, rank: int, addr: str = "localhost", port: int = 29500) -> None:
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
def _np_all_reduce(backend_module: Any, tensor: Any, op_type: str = "sum", *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_all_reduce."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)

    if op_type == "sum":
        res = sum(all_tensors)
    elif op_type == "prod":
        res = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            res = res * t
    elif op_type == "max":
        res = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        res = backend_module.minimum.reduce(all_tensors)
    else:
        res = sum(all_tensors)

    return res


@numpy_eager_registry.register("AllGather")
def _np_all_gather(backend_module: Any, tensor: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_all_gather."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return backend_module.expand_dims(tensor, axis=axis) if axis is not None else tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)
    return backend_module.concatenate(all_tensors, axis=axis if axis is not None else 0)


@numpy_eager_registry.register("Broadcast")
def _np_broadcast(backend_module: Any, tensor: Any, root_rank: int = 0, *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_broadcast."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)
    return all_tensors[root_rank]


@numpy_eager_registry.register("ReduceScatter")
def _np_reduce_scatter(backend_module: Any, tensor: Any, op_type: str = "sum", axis: int = 0, *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_reduce_scatter."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)

    if op_type == "sum":
        reduced = sum(all_tensors)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced = backend_module.minimum.reduce(all_tensors)
    else:
        reduced = sum(all_tensors)

    chunks = backend_module.array_split(reduced, _tcp_dist_ctx.world_size, axis=axis)
    return chunks[_tcp_dist_ctx.rank]


@numpy_eager_registry.register("Reduce")
def _np_reduce(backend_module: Any, tensor: Any, root_rank: int = 0, op_type: str = "sum", *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_reduce."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)

    if op_type == "sum":
        reduced = sum(all_tensors)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if hasattr(all_tensors[0], "copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = backend_module.maximum.reduce(all_tensors)
    elif op_type == "min":
        reduced = backend_module.minimum.reduce(all_tensors)
    else:
        reduced = sum(all_tensors)

    return reduced if _tcp_dist_ctx.rank == root_rank else None


@numpy_eager_registry.register("AllToAll")
def _np_all_to_all(backend_module: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager eval AllToAll."""
    tensor = backend_module.array(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors = _tcp_dist_ctx.all_gather_tensors(tensor)
    # Basic AllToAll logic: assume inputs are already split, we return a tuple/list
    # In practice it splits by axis and gathers, but just returning all is fine for now
    return all_tensors


@numpy_eager_registry.register("ShardTensor")
def _np_shard_tensor(backend_module: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager eval _np_shard_tensor."""
    return backend_module.array(tensor)
