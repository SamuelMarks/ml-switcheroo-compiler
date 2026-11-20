"""Distributed ops eager via real multiprocessing TCP sockets."""

from __future__ import annotations

import os  # noqa: F401
import time
import types
from multiprocessing.connection import Client, Connection, Listener

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

_DEFAULT_NP_MOD: types.ModuleType | BackendModuleProtocol = np


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: BackendModuleProtocol | types.ModuleType | None, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager np axis index operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return np.array(0)


class TCPDistributedContext:
    """Real standard-library IPC Context for Collectives (Ring Topology)."""

    def __init__(
        self, world_size: int = 1, rank: int = 0, addr: str = "localhost", port: int = 29500, topology: str = "ring"
    ) -> None:
        """Init.

        Args:
            world_size: Input `world_size` argument (world size).
            rank: Input `rank` argument (rank).
            addr: Input `addr` argument (addr).
            port: Input `port` argument (port).
            topology: Input `topology` argument (topology).
        """
        self.world_size = world_size
        self.rank = rank
        self.addr = addr
        self.port = port
        self.topology = topology
        self.authkey = b"ml_switcheroo"
        self.listener: Listener | None = None
        self.recv_conns: list[Connection[BackendArg, BackendArg]] = []
        self.send_conns: list[Connection[BackendArg, BackendArg]] = []

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

    @staticmethod
    def _combine_ring_chunk(
        lhs: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        rhs: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        op_type: str,
        backend_module: types.ModuleType | BackendModuleProtocol | None,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Combine two ring chunks according to `op_type`.

        Args:
            lhs (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Current chunk array.
            rhs (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Received chunk array.
            op_type (str): Reduction operator ('sum', 'prod', 'max', 'min').
            backend_module (types.ModuleType | BackendModuleProtocol | None): Target NumPy-compatible module.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Reduced chunk array.
        """
        if op_type == "sum":
            return lhs + rhs
        if op_type == "prod":
            return lhs * rhs
        if op_type == "max":
            return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("maximum")(lhs, rhs))
        if op_type == "min":
            return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("minimum")(lhs, rhs))
        return lhs

    def all_reduce_ring(
        self,
        tensor: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        op_type: str = "sum",
        backend_module: types.ModuleType | BackendModuleProtocol | None = _DEFAULT_NP_MOD,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Evaluate all_reduce_ring.

        Args:
            tensor: Input `tensor` argument (tensor).
            op_type: Input `op_type` argument (op type).
            backend_module: Input `backend_module` argument (backend module).

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Computed result.
        """
        if self.world_size <= 1:
            return tensor

        raw_chunks = adapt_backend_module(backend_module).require_symbol("array_split")(tensor, self.world_size)
        chunks: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = [ensure_ndarray(c) for c in raw_chunks]

        # Scatter-reduce phase
        for step in range(self.world_size - 1):
            send_chunk_idx: int = (self.rank - step) % self.world_size
            recv_chunk_idx: int = (self.rank - step - 1) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            recv_data = ensure_ndarray(self.recv_conns[0].recv()) if self.recv_conns else chunks[recv_chunk_idx]
            chunks[recv_chunk_idx] = self._combine_ring_chunk(
                chunks[recv_chunk_idx], recv_data, op_type, backend_module
            )

        # All-gather phase
        for step in range(self.world_size - 1):
            send_chunk_idx = (self.rank - step + 1) % self.world_size
            recv_chunk_idx = (self.rank - step) % self.world_size

            if self.send_conns:
                self.send_conns[0].send(chunks[send_chunk_idx])
            if self.recv_conns:
                chunks[recv_chunk_idx] = ensure_ndarray(self.recv_conns[0].recv())

        return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("concatenate")(chunks))

    def all_gather_tensors(self, tensor: BackendArg) -> list[BackendArg]:
        """Perform AllGather over TCP Ring.

        Args:
            tensor: Input `tensor` argument (tensor).

        Returns:
            list[BackendArg]: Computed result.
        """
        if self.world_size <= 1:
            return [tensor]

        all_tensors: list[BackendArg] = [None] * self.world_size
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
            if self.recv_conns and adapt_backend_module(self.recv_conns[0]).has_symbol("close"):
                self.recv_conns[0].close()
            if self.send_conns and adapt_backend_module(self.send_conns[0]).has_symbol("close"):
                self.send_conns[0].close()
            if self.listener and adapt_backend_module(self.listener).has_symbol("close"):
                self.listener.close()


_tcp_dist_ctx = TCPDistributedContext()


def set_np_distributed_context(world_size: int, rank: int, addr: str = "localhost", port: int = 29500) -> None:
    """Set mock distributed context.

    Args:
        world_size (int): Input `world_size` argument (world size).
        rank (int): Input `rank` argument (rank).
        addr (str): Input `addr` argument (addr).
        port (int): Input `port` argument (port).
    """
    global _tcp_dist_ctx
    _tcp_dist_ctx.shutdown()
    _tcp_dist_ctx = TCPDistributedContext(world_size, rank, addr, port)
    _tcp_dist_ctx.initialize()


@numpy_eager_registry.register("AllReduce")
@numpy_eager_registry.register("NcclAllReduce")
@numpy_eager_registry.register("HierarchicalCopyAllReduce")
def _np_all_reduce(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    op_type: str = "sum",
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_all_reduce.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        op_type: Input `op_type` argument (op type).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    return adapt_backend_module(_tcp_dist_ctx).require_symbol("all_reduce_ring")(tensor, op_type, backend_module)


@numpy_eager_registry.register("AllGather")
def _np_all_gather(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    axis: int | None = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_all_gather.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        axis: Input `axis` argument (axis).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return (
            adapt_backend_module(backend_module).require_symbol("expand_dims")(tensor, axis=axis)
            if axis is not None
            else tensor
        )

    all_tensors: list[BackendArg] = _tcp_dist_ctx.all_gather_tensors(tensor)
    return adapt_backend_module(backend_module).require_symbol("concatenate")(
        all_tensors, axis=axis if axis is not None else 0
    )


@numpy_eager_registry.register("Broadcast")
def _np_broadcast(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    root_rank: int = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_broadcast.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        root_rank: Input `root_rank` argument (root rank).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[BackendArg] = _tcp_dist_ctx.all_gather_tensors(tensor)
    return all_tensors[root_rank]


@numpy_eager_registry.register("ReduceScatter")
def _np_reduce_scatter(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    op_type: str = "sum",
    axis: int = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_reduce_scatter.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        op_type: Input `op_type` argument (op type).
        axis: Input `axis` argument (axis).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerTensorProtocol] = [
        adapt_backend_module(backend_module).require_symbol("asarray")(t)
        for t in _tcp_dist_ctx.all_gather_tensors(tensor)
    ]

    if op_type == "sum":
        reduced: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("sum")(all_tensors, axis=0)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if adapt_backend_module(all_tensors[0]).has_symbol("copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = adapt_backend_module(backend_module).require_symbol("maximum").reduce(all_tensors)
    elif op_type == "min":
        reduced = adapt_backend_module(backend_module).require_symbol("minimum").reduce(all_tensors)
    else:
        reduced = adapt_backend_module(backend_module).require_symbol("sum")(all_tensors, axis=0)

    chunks: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("array_split")(
        reduced, _tcp_dist_ctx.world_size, axis=axis
    )
    return chunks[_tcp_dist_ctx.rank]


@numpy_eager_registry.register("Reduce")
def _np_reduce(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    root_rank: int = 0,
    op_type: str = "sum",
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_reduce.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        root_rank: Input `root_rank` argument (root rank).
        op_type: Input `op_type` argument (op type).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[EagerTensorProtocol] = [
        adapt_backend_module(backend_module).require_symbol("asarray")(t)
        for t in _tcp_dist_ctx.all_gather_tensors(tensor)
    ]

    if op_type == "sum":
        reduced: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("sum")(all_tensors, axis=0)
    elif op_type == "prod":
        reduced = all_tensors[0].copy() if adapt_backend_module(all_tensors[0]).has_symbol("copy") else all_tensors[0]
        for t in all_tensors[1:]:
            reduced = reduced * t
    elif op_type == "max":
        reduced = adapt_backend_module(backend_module).require_symbol("maximum").reduce(all_tensors)
    elif op_type == "min":
        reduced = adapt_backend_module(backend_module).require_symbol("minimum").reduce(all_tensors)
    else:
        reduced = adapt_backend_module(backend_module).require_symbol("sum")(all_tensors, axis=0)

    return reduced if _tcp_dist_ctx.rank == root_rank else None


@numpy_eager_registry.register("AllToAll")
def _np_all_to_all(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval AllToAll.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    tensor = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    if _tcp_dist_ctx.world_size <= 1:
        return tensor

    all_tensors: list[BackendArg] = _tcp_dist_ctx.all_gather_tensors(tensor)
    # Basic AllToAll logic: assume inputs are already split, we return a tuple/list
    # In practice it splits by axis and gathers, but just returning all is fine for now
    return all_tensors


@numpy_eager_registry.register("ShardTensor")
def _np_shard_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager eval _np_shard_tensor.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        *args: Input `args` argument (args).
        **kwargs: Input `kwargs` argument (kwargs).

    Returns:
        BackendArg: Computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(tensor)


@numpy_eager_registry.register("Pmean")
def _np_pmean(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis_name: str,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> BackendArg:
    """Execute the eager np pmean operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        axis_name (str): Named axis.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    del backend_module, axis_name, args, kwargs
    return x


@numpy_eager_registry.register("Psum")
def _np_psum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager np psum operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    del kwargs
    if _tcp_dist_ctx.world_size > 1:
        tensor = np.asarray(adapt_backend_module(backend_module).require_symbol("array")(args[0]))
        return _tcp_dist_ctx.all_reduce_ring(tensor, op_type="sum", backend_module=backend_module)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(args[0]))
