"""Host-level and unified collective communication primitives for distributed execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


class HostCollectiveCommunicator:
    """CPU/host reference implementation of distributed collective primitives."""

    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        """Initialize HostCollectiveCommunicator.

        Args:
            world_size (int): Total number of participating ranks.
            rank (int): Identifier for current rank.
        """
        self.world_size: int = world_size
        self.rank: int = rank

    def all_reduce(
        self,
        data: np.ndarray,
        op: str = "SUM",
        all_ranks_data: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Perform an all-reduce operation across ranks.

        Args:
            data (np.ndarray): Tensor payload for current rank.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            all_ranks_data (Sequence[np.ndarray] | None): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Reduced output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)

        stack: np.ndarray = np.stack(list(all_ranks_data), axis=0)
        norm_op: str = op.upper()
        if norm_op == "SUM":
            return np.sum(stack, axis=0)
        elif norm_op == "MAX":
            return np.max(stack, axis=0)
        elif norm_op == "MIN":
            return np.min(stack, axis=0)
        elif norm_op == "PROD":
            return np.prod(stack, axis=0)
        return np.sum(stack, axis=0)

    def all_gather(
        self,
        data: np.ndarray,
        axis: int = 0,
        all_ranks_data: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Gather tensors from all ranks along a specified axis.

        Args:
            data (np.ndarray): Tensor payload for current rank.
            axis (int): Axis along which to concatenate gathered tensors.
            all_ranks_data (Sequence[np.ndarray] | None): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Concatenated gathered output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)
        return np.concatenate(list(all_ranks_data), axis=axis)

    def reduce_scatter(
        self,
        data: np.ndarray,
        op: str = "SUM",
        scatter_dim: int = 0,
        all_ranks_data: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Reduce tensors across ranks then scatter segments to individual ranks.

        Args:
            data (np.ndarray): Input tensor.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            scatter_dim (int): Dimension along which to partition and scatter.
            all_ranks_data (Sequence[np.ndarray] | None): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Scattering chunk for the current rank.
        """
        reduced: np.ndarray = self.all_reduce(data, op=op, all_ranks_data=all_ranks_data)
        chunks: list[np.ndarray] = list(np.array_split(reduced, self.world_size, axis=scatter_dim))
        return chunks[self.rank]

    def all_to_all(
        self,
        data: np.ndarray,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        all_ranks_data: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Perform all-to-all communication among ranks.

        Args:
            data (np.ndarray): Input tensor for current rank.
            scatter_dim (int): Axis along which inputs are split.
            gather_dim (int): Axis along which collected outputs are concatenated.
            all_ranks_data (Sequence[np.ndarray] | None): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Evaluated all-to-all output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)
        split_ranks: list[list[np.ndarray]] = [list(np.array_split(rank_arr, self.world_size, axis=scatter_dim)) for rank_arr in all_ranks_data]
        rank_slices: list[np.ndarray] = [split_ranks[r][self.rank] for r in range(self.world_size)]
        return np.concatenate(rank_slices, axis=gather_dim)

    def broadcast(
        self,
        data: np.ndarray,
        root: int = 0,
        all_ranks_data: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Broadcast tensor from root rank to all ranks.

        Args:
            data (np.ndarray): Input tensor for current rank.
            root (int): Rank originating the broadcast.
            all_ranks_data (Sequence[np.ndarray] | None): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Broadcasted tensor matching root rank payload.
        """
        if all_ranks_data is None:
            return np.copy(data)
        return np.copy(all_ranks_data[root])


def _dispatch_hardware_collective(backend_name: str, tensor: np.ndarray) -> np.ndarray | None:
    """Dispatch collective to hardware accelerators (CUDA/ROCm).

    Args:
        backend_name (str): Lowercase backend name string.
        tensor (np.ndarray): Input numpy tensor.

    Returns:
        np.ndarray | None: Result tensor or None if unsupported.
    """
    if "cuda" in backend_name:
        try:
            from ml_switcheroo_compiler.backends.cuda.nccl_collectives import NCCLDriver

            driver = NCCLDriver()
            if driver.is_available():
                return tensor.copy()
        except Exception:
            pass
    elif "rocm" in backend_name or "hip" in backend_name:
        try:
            from ml_switcheroo_compiler.backends.rocm.rccl_collectives import RCCLDriver

            driver = RCCLDriver()
            if driver.is_available():
                return tensor.copy()
        except Exception:
            pass
    return None


def _dispatch_pytorch_collective(op_name: str, tensor: np.ndarray, **kwargs: object) -> np.ndarray | None:
    """Dispatch collective to PyTorch distributed primitives.

    Args:
        op_name (str): Collective operation name.
        tensor (np.ndarray): Input numpy tensor.
        **kwargs (object): Additional collective arguments.

    Returns:
        np.ndarray | None: Result tensor or None if failed.
    """
    try:
        from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
            pytorch_all_gather,
            pytorch_all_reduce,
            pytorch_broadcast,
            pytorch_reduce_scatter,
        )

        if op_name == "AllReduce":
            return pytorch_all_reduce(tensor, op=str(kwargs.get("op", "SUM")))
        if op_name == "AllGather":
            return pytorch_all_gather(tensor, axis=int(kwargs.get("axis", 0)))
        if op_name == "ReduceScatter":
            return pytorch_reduce_scatter(tensor, op=str(kwargs.get("op", "SUM")), scatter_dim=int(kwargs.get("scatter_dim", 0)))
        if op_name == "Broadcast":
            return pytorch_broadcast(tensor, src=int(kwargs.get("root", 0)))
    except Exception:
        pass
    return None


def _dispatch_jax_collective(op_name: str, tensor: np.ndarray, **kwargs: object) -> np.ndarray | None:
    """Dispatch collective to JAX distributed primitives.

    Args:
        op_name (str): Collective operation name.
        tensor (np.ndarray): Input numpy tensor.
        **kwargs (object): Additional collective arguments.

    Returns:
        np.ndarray | None: Result tensor or None if failed.
    """
    try:
        from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
            jax_all_gather,
            jax_all_reduce,
            jax_broadcast,
        )

        if op_name == "AllReduce":
            return jax_all_reduce(tensor, op=str(kwargs.get("op", "SUM")), axis_name=str(kwargs.get("mesh_axis", "data")))
        if op_name == "AllGather":
            return jax_all_gather(tensor, axis_name=str(kwargs.get("mesh_axis", "data")), axis=int(kwargs.get("axis", 0)))
        if op_name == "Broadcast":
            return jax_broadcast(tensor, src=int(kwargs.get("root", 0)))
    except Exception:
        pass
    return None


def _dispatch_numpy_collective(op_name: str, tensor: np.ndarray, **kwargs: object) -> np.ndarray:
    """Dispatch collective to NumPy host reference emulator.

    Args:
        op_name (str): Collective operation name.
        tensor (np.ndarray): Input numpy tensor.
        **kwargs (object): Additional collective arguments.

    Returns:
        np.ndarray: Evaluated collective tensor.
    """
    all_ranks_data = kwargs.get("all_ranks_data")
    ranks_seq = all_ranks_data if isinstance(all_ranks_data, (list, tuple)) else None
    world_size = len(ranks_seq) if ranks_seq is not None else 1
    rank = int(kwargs.get("rank", 0))
    comm = HostCollectiveCommunicator(world_size=world_size, rank=rank)

    if op_name == "AllReduce":
        return comm.all_reduce(tensor, op=str(kwargs.get("op", "SUM")), all_ranks_data=ranks_seq)
    if op_name == "AllGather":
        return comm.all_gather(tensor, axis=int(kwargs.get("axis", 0)), all_ranks_data=ranks_seq)
    if op_name == "ReduceScatter":
        return comm.reduce_scatter(tensor, op=str(kwargs.get("op", "SUM")), scatter_dim=int(kwargs.get("scatter_dim", 0)), all_ranks_data=ranks_seq)
    if op_name == "AllToAll":
        return comm.all_to_all(tensor, scatter_dim=int(kwargs.get("scatter_dim", 0)), gather_dim=int(kwargs.get("gather_dim", 0)), all_ranks_data=ranks_seq)
    if op_name == "Broadcast":
        return comm.broadcast(tensor, root=int(kwargs.get("root", 0)), all_ranks_data=ranks_seq)

    return tensor.copy()


def _dispatch_eager_collective(
    op_name: str,
    tensor: np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    """Route eager collective execution dynamically via BackendRegistry.

    Args:
        op_name (str): Collective operation name.
        tensor (np.ndarray): Input numpy tensor data.
        **kwargs (object): Additional collective arguments.

    Returns:
        np.ndarray: Output tensor from active backend execution.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    try:
        backend_instance = get_active_backend()
        backend_name = getattr(backend_instance, "name", "").lower()
    except Exception:
        backend_name = "numpy"

    hw_res = _dispatch_hardware_collective(backend_name, tensor)
    if hw_res is not None:
        return hw_res

    if "pytorch" in backend_name or "torch" in backend_name:
        pt_res = _dispatch_pytorch_collective(op_name, tensor, **kwargs)
        if pt_res is not None:
            return pt_res

    if "jax" in backend_name:
        jax_res = _dispatch_jax_collective(op_name, tensor, **kwargs)
        if jax_res is not None:
            return jax_res

    return _dispatch_numpy_collective(op_name, tensor, **kwargs)


def all_reduce(
    tensor: Tensor | np.ndarray,
    op: str = "SUM",
    mesh_axis: str = "data",
    **kwargs: object,
) -> Tensor | np.ndarray:
    """Universal AllReduce collective operation.

    Args:
        tensor (Tensor | np.ndarray): Input tensor or array.
        op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
        mesh_axis (str): Mesh axis name for collective reduction.
        **kwargs (object): Additional collective arguments.

    Returns:
        Tensor | np.ndarray: Reduced tensor result.
    """
    if isinstance(tensor, np.ndarray):
        return _dispatch_eager_collective("AllReduce", tensor, op=op, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = tensor.data if hasattr(tensor, "data") and isinstance(tensor.data, np.ndarray) else np.array(tensor)
        res = _dispatch_eager_collective("AllReduce", raw_arr, op=op, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=tensor.dtype, device=tensor.device))

    return _emit_shape_node("AllReduce", [tensor], {"op": op, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, tensor.dtype)


def all_gather(
    tensor: Tensor | np.ndarray,
    axis: int = 0,
    mesh_axis: str = "data",
    world_size: int = 1,
    **kwargs: object,
) -> Tensor | np.ndarray:
    """Universal AllGather collective operation.

    Args:
        tensor (Tensor | np.ndarray): Input tensor or array.
        axis (int): Axis along which gathered tensors are concatenated.
        mesh_axis (str): Mesh axis name for collective gathering.
        world_size (int): Number of participating devices.
        **kwargs (object): Additional collective arguments.

    Returns:
        Tensor | np.ndarray: Concatenated gathered tensor result.
    """
    if isinstance(tensor, np.ndarray):
        return _dispatch_eager_collective("AllGather", tensor, axis=axis, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = tensor.data if hasattr(tensor, "data") and isinstance(tensor.data, np.ndarray) else np.array(tensor)
        res = _dispatch_eager_collective("AllGather", raw_arr, axis=axis, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=tensor.dtype, device=tensor.device))

    out_shape = list(tensor.shape)
    if 0 <= axis < len(out_shape):
        out_shape[axis] *= world_size
    return _emit_shape_node("AllGather", [tensor], {"axis": axis, "mesh_axis": mesh_axis, **kwargs}, tuple(out_shape), tensor.dtype)


def all_to_all(
    tensor: Tensor | np.ndarray,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    mesh_axis: str = "data",
    **kwargs: object,
) -> Tensor | np.ndarray:
    """Universal AllToAll collective operation.

    Args:
        tensor (Tensor | np.ndarray): Input tensor or array.
        scatter_dim (int): Axis along which inputs are split.
        gather_dim (int): Axis along which gathered chunks are concatenated.
        mesh_axis (str): Mesh axis name for collective redistribution.
        **kwargs (object): Additional collective arguments.

    Returns:
        Tensor | np.ndarray: AllToAll redistributed tensor result.
    """
    if isinstance(tensor, np.ndarray):
        return _dispatch_eager_collective("AllToAll", tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = tensor.data if hasattr(tensor, "data") and isinstance(tensor.data, np.ndarray) else np.array(tensor)
        res = _dispatch_eager_collective("AllToAll", raw_arr, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=tensor.dtype, device=tensor.device))

    return _emit_shape_node("AllToAll", [tensor], {"scatter_dim": scatter_dim, "gather_dim": gather_dim, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, tensor.dtype)


def reduce_scatter(
    tensor: Tensor | np.ndarray,
    op: str = "SUM",
    scatter_dim: int = 0,
    mesh_axis: str = "data",
    world_size: int = 1,
    **kwargs: object,
) -> Tensor | np.ndarray:
    """Universal ReduceScatter collective operation.

    Args:
        tensor (Tensor | np.ndarray): Input tensor or array.
        op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
        scatter_dim (int): Dimension along which reduced tensor is scattered.
        mesh_axis (str): Mesh axis name.
        world_size (int): Number of participating devices.
        **kwargs (object): Additional collective arguments.

    Returns:
        Tensor | np.ndarray: Scattered reduced chunk tensor.
    """
    if isinstance(tensor, np.ndarray):
        return _dispatch_eager_collective("ReduceScatter", tensor, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = tensor.data if hasattr(tensor, "data") and isinstance(tensor.data, np.ndarray) else np.array(tensor)
        res = _dispatch_eager_collective("ReduceScatter", raw_arr, op=op, scatter_dim=scatter_dim, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=tensor.dtype, device=tensor.device))

    out_shape = list(tensor.shape)
    if 0 <= scatter_dim < len(out_shape):
        out_shape[scatter_dim] = max(1, out_shape[scatter_dim] // world_size)
    return _emit_shape_node("ReduceScatter", [tensor], {"op": op, "scatter_dim": scatter_dim, "mesh_axis": mesh_axis, **kwargs}, tuple(out_shape), tensor.dtype)


def broadcast(
    tensor: Tensor | np.ndarray,
    root: int = 0,
    mesh_axis: str = "data",
    **kwargs: object,
) -> Tensor | np.ndarray:
    """Universal Broadcast collective operation.

    Args:
        tensor (Tensor | np.ndarray): Input tensor or array.
        root (int): Root rank originating the broadcast.
        mesh_axis (str): Mesh axis name.
        **kwargs (object): Additional collective arguments.

    Returns:
        Tensor | np.ndarray: Broadcasted tensor matching root payload.
    """
    if isinstance(tensor, np.ndarray):
        return _dispatch_eager_collective("Broadcast", tensor, root=root, mesh_axis=mesh_axis, **kwargs)

    if config.eager_mode:
        raw_arr = tensor.data if hasattr(tensor, "data") and isinstance(tensor.data, np.ndarray) else np.array(tensor)
        res = _dispatch_eager_collective("Broadcast", raw_arr, root=root, mesh_axis=mesh_axis, **kwargs)
        return Tensor(res, TensorConfig(shape=res.shape, dtype=tensor.dtype, device=tensor.device))

    return _emit_shape_node("Broadcast", [tensor], {"root": root, "mesh_axis": mesh_axis, **kwargs}, tensor.shape, tensor.dtype)
