"""Zero-copy collective operations for distributed execution over WebRTC."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    _apply_reduction_op as _apply_reduction_op,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    all_gather as all_gather,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    all_reduce as all_reduce,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    broadcast as broadcast,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    recursive_halving_doubling_all_gather as recursive_halving_doubling_all_gather,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_ring_ag import (
    ring_all_reduce as ring_all_reduce,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_scatter_barrier import (
    DistributedBarrier as DistributedBarrier,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_scatter_barrier import (
    all_to_all as all_to_all,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_scatter_barrier import (
    reduce_scatter as reduce_scatter,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_scatter_barrier import (
    ring_reduce_scatter as ring_reduce_scatter,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    HIGH_WATERMARK_BYTES as HIGH_WATERMARK_BYTES,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    LOW_WATERMARK_BYTES as LOW_WATERMARK_BYTES,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    WebRTCDataChannelStream as WebRTCDataChannelStream,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _as_tensor_seq as _as_tensor_seq,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _BufferedLowJsProto as _BufferedLowJsProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _BufferedLowPyProto as _BufferedLowPyProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _OnBufferedLowJsProto as _OnBufferedLowJsProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _OnBufferedLowPyProto as _OnBufferedLowPyProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _OnMessageJsProto as _OnMessageJsProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _OnMessagePyProto as _OnMessagePyProto,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _recv_data as _recv_data,
)
from ml_switcheroo_compiler.backends.edge.distributed_webrtc._collectives_stream import (
    _send_data as _send_data,
)
from ml_switcheroo_compiler.backends.registry import get_active_backend

_BACKEND_COLLECTIVE_REGISTRY: dict[str, BackendArg] = {}


def register_collective_backend(backend_name: str, dispatcher: BackendArg) -> None:
    """Register a backend-specific collective dispatcher.

    Args:
        backend_name (str): The name of the backend (e.g. 'pytorch', 'jax', 'cuda').
        dispatcher (BackendArg): Dispatcher callable or instance.
    """
    _BACKEND_COLLECTIVE_REGISTRY[backend_name.lower()] = dispatcher


def get_collective_backend(backend_name: str) -> BackendArg | None:
    """Retrieve a registered collective dispatcher for a backend.

    Args:
        backend_name (str): Target backend name.

    Returns:
        Optional[object]: Registered collective dispatcher or None.
    """
    return _BACKEND_COLLECTIVE_REGISTRY.get(backend_name.lower())


def _dispatch_pytorch_collective(op: str, tensor: BackendArg, kwargs: dict[str, BackendArg]) -> BackendArg | None:
    """Attempt to dispatch collective operation to PyTorch distributed backend via BackendRegistry.

    Args:
        op (str): Collective operation name.
        tensor (BackendArg): Local input tensor.
        kwargs (dict[str, BackendArg]): Additional collective parameters.

    Returns:
        BackendArg | None: Computed tensor if handled, else None.
    """
    try:
        from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        pt_adp = adapt_backend_module(BackendRegistry.get("pytorch"))
        group = kwargs.get("group")
        if op == "AllReduce":
            return pt_adp.require_op("all_reduce")(tensor, op=str(kwargs.get("op_type", "SUM")), group=group)
        if op == "AllGather":
            return pt_adp.require_op("all_gather")(tensor, axis=int(coerce_int(kwargs.get("axis", 0))), group=group)
        if op == "ReduceScatter":
            return pt_adp.require_op("reduce_scatter")(
                tensor,
                op=str(kwargs.get("op_type", "SUM")),
                scatter_dim=int(coerce_int(kwargs.get("scatter_dim", 0))),
                group=group,
            )
        if op == "Broadcast":
            return pt_adp.require_op("broadcast")(tensor, src=int(coerce_int(kwargs.get("root", 0))), group=group)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _dispatch_jax_collective(op: str, tensor: BackendArg, kwargs: dict[str, BackendArg]) -> BackendArg | None:
    """Attempt to dispatch collective operation to JAX distributed backend via BackendRegistry.

    Args:
        op (str): Collective operation name.
        tensor (BackendArg): Local input tensor.
        kwargs (dict[str, BackendArg]): Additional collective parameters.

    Returns:
        BackendArg | None: Computed tensor if handled, else None.
    """
    try:
        from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        jax_adp = adapt_backend_module(BackendRegistry.get("jax"))
        axis_name = str(kwargs.get("axis_name", "data"))
        if op == "AllReduce":
            return jax_adp.require_op("all_reduce")(tensor, op=str(kwargs.get("op_type", "SUM")), axis_name=axis_name)
        if op == "AllGather":
            return jax_adp.require_op("all_gather")(
                tensor, axis_name=axis_name, axis=int(coerce_int(kwargs.get("axis", 0)))
            )
        if op == "Broadcast":
            return jax_adp.require_op("broadcast")(tensor, src=int(coerce_int(kwargs.get("root", 0))))
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _reduce_rank_sequence(all_ranks_data: Sequence[BackendArg], red_op: str) -> BackendArg:
    """Reduce a sequence of tensors across all ranks using the specified operator.

    Args:
        all_ranks_data (Sequence[BackendArg]): Tensors from all participating ranks.
        red_op (str): Uppercase reduction operator string.

    Returns:
        BackendArg: Reduced result tensor.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    backend = get_active_backend()
    accum: BackendArg = adapt_backend_module(all_ranks_data[0]).require_op("copy")()
    for other in all_ranks_data[1:]:
        other_t = coerce_numeric(other)
        if red_op in ("SUM", "ADD"):
            accum = coerce_numeric(accum) + other_t
        elif red_op in ("PROD", "PRODUCT"):
            accum = coerce_numeric(accum) * other_t
        elif red_op == "MAX":
            accum = backend.execute_op("maximum", accum, other_t)
        elif red_op == "MIN":
            accum = backend.execute_op("minimum", accum, other_t)
    return accum


def _simulate_multi_rank_collective(
    op: str,
    all_ranks_data: Sequence[BackendArg],
    kwargs: dict[str, BackendArg],
) -> BackendArg | None:
    """Simulate a collective operation across multi-rank data on the host.

    Args:
        op (str): Collective operation name.
        all_ranks_data (Sequence[BackendArg]): List or tuple of tensors for all ranks.
        kwargs (dict[str, BackendArg]): Additional collective parameters.

    Returns:
        BackendArg | None: Simulated collective result if op is recognized, else None.
    """
    backend = get_active_backend()
    world_size = len(all_ranks_data)
    rank = int(coerce_int(kwargs.get("rank", 0)))
    red_op = str(kwargs.get("op_type", "SUM")).upper()

    if op == "AllReduce":
        return _reduce_rank_sequence(all_ranks_data, red_op)
    if op == "AllGather":
        return backend.execute_op("concatenate", all_ranks_data, axis=int(coerce_int(kwargs.get("axis", 0))))
    if op == "ReduceScatter":
        accum = _reduce_rank_sequence(all_ranks_data, red_op)
        chunks = coerce_operand_list(
            backend.execute_op("array_split", accum, world_size, axis=int(coerce_int(kwargs.get("scatter_dim", 0))))
        )
        return chunks[rank]
    if op == "AllToAll":
        scatter_dim = int(coerce_int(kwargs.get("scatter_dim", 0)))
        gather_dim = int(coerce_int(kwargs.get("gather_dim", 0)))
        gathered = [
            coerce_operand_list(backend.execute_op("array_split", all_ranks_data[src], world_size, axis=scatter_dim))[
                rank
            ]
            for src in range(world_size)
        ]
        return backend.execute_op("concatenate", gathered, axis=gather_dim)
    if op == "Broadcast":
        from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

        return adapt_backend_module(all_ranks_data[int(coerce_int(kwargs.get("root", 0)))]).require_op("copy")()
    return None


def dispatch_collective(
    op: str,
    tensor: BackendArg,
    backend: str = "numpy",
    **kwargs: BackendArg,
) -> BackendArg:
    """Dispatch a collective operation to the designated backend runtime.

    Args:
        op (str): Collective operation name ("AllReduce", "AllGather", "ReduceScatter", "AllToAll", "Broadcast").
        tensor (BackendArg): Local tensor data.
        backend (str): Target execution backend ("pytorch", "jax", "cuda", "numpy").
        **kwargs (BackendArg): Additional collective parameters (op_type, axis, rank, world_size, all_ranks_data).

    Returns:
        BackendArg: Evaluated collective result.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    backend_key = backend.lower()
    custom_disp = get_collective_backend(backend_key)
    if custom_disp is not None and callable(custom_disp):
        return custom_disp(op, tensor, **kwargs)

    res: BackendArg | None = None
    if backend_key in ("pytorch", "torch"):
        res = _dispatch_pytorch_collective(op, tensor, kwargs)
    elif backend_key == "jax":
        res = _dispatch_jax_collective(op, tensor, kwargs)

    if res is not None:
        return res

    all_ranks_data = kwargs.get("all_ranks_data")
    if isinstance(all_ranks_data, (list, tuple)) and len(all_ranks_data) > 0:
        sim_res = _simulate_multi_rank_collective(op, all_ranks_data, kwargs)
        if sim_res is not None:
            return sim_res

    return adapt_backend_module(tensor).require_op("copy")()
