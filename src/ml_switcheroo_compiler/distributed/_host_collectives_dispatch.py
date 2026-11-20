"""Host reference and active-backend eager collective dispatchers."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import ml_switcheroo_compiler.distributed.host_collectives as _hc_mod
from ml_switcheroo_compiler.distributed._host_collectives_communicator import HostCollectiveCommunicator
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    TensorLike,
    _copy_if_possible,
    _hc_to_int,
)


@typing.runtime_checkable
class _NamedBackendProto(typing.Protocol):
    """Protocol for backend classes exposing `.name`."""

    @property
    def name(self) -> str:
        """Return backend identifier string.

        Returns:
            str: Backend name.
        """
        ...


@typing.runtime_checkable
class _CollectiveBackendProto(typing.Protocol):
    """Protocol for backend classes exposing `execute_collective`."""

    def execute_collective(
        self,
        op_name: str,
        tensor: TensorLike,
        **kwargs: int | float | str | Sequence[TensorLike] | None,
    ) -> TensorLike:
        """Execute collective operation on the active backend.

        Args:
            op_name (str): Collective operation name.
            tensor (TensorLike): Input tensor buffer.
            **kwargs: Collective configuration keyword arguments.

        Returns:
            TensorLike: Collective output tensor or `None`.
        """
        ...


def _dispatch_host_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike:
    """Dispatch collective to host reference emulator.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike: Evaluated collective tensor.
    """
    all_ranks_data = kwargs.get("all_ranks_data")
    ranks_seq = all_ranks_data if isinstance(all_ranks_data, (list, tuple)) else None
    world_size = len(ranks_seq) if ranks_seq is not None else 1
    rank = _hc_to_int(kwargs.get("rank", 0), 0)
    comm = HostCollectiveCommunicator(world_size=world_size, rank=rank)

    if op_name == "AllReduce":
        return comm.all_reduce(tensor, op=str(kwargs.get("op", "SUM")), all_ranks_data=ranks_seq)
    if op_name == "AllGather":
        return comm.all_gather(tensor, axis=_hc_to_int(kwargs.get("axis", 0), 0), all_ranks_data=ranks_seq)
    if op_name == "ReduceScatter":
        return comm.reduce_scatter(
            tensor,
            op=str(kwargs.get("op", "SUM")),
            scatter_dim=_hc_to_int(kwargs.get("scatter_dim", 0), 0),
            all_ranks_data=ranks_seq,
        )
    if op_name == "AllToAll":
        return comm.all_to_all(
            tensor,
            scatter_dim=_hc_to_int(kwargs.get("scatter_dim", 0), 0),
            gather_dim=_hc_to_int(kwargs.get("gather_dim", 0), 0),
            all_ranks_data=ranks_seq,
        )
    if op_name == "Broadcast":
        return comm.broadcast(tensor, root=_hc_to_int(kwargs.get("root", 0), 0), all_ranks_data=ranks_seq)

    return _copy_if_possible(tensor)


def _dispatch_eager_collective(
    op_name: str,
    tensor: TensorLike,
    **kwargs: int | float | str | Sequence[TensorLike] | None,
) -> TensorLike:
    """Route eager collective execution dynamically via BackendRegistry protocol methods.

    Args:
        op_name (str): Collective operation name.
        tensor (TensorLike): Input tensor buffer.
        **kwargs: Additional collective arguments.

    Returns:
        TensorLike: Output tensor from active backend execution or host collective emulator.
    """
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    raw_inst: type[BaseGenerator] | _CollectiveBackendProto | _NamedBackendProto | None = None
    try:
        active_cls: type[BaseGenerator] | _NamedBackendProto = get_active_backend()
        raw_inst = active_cls
        backend_name = str(raw_inst.name).lower() if isinstance(raw_inst, _NamedBackendProto) else ""
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        raw_inst = None
        backend_name = ""

    hw_res = _hc_mod._dispatch_accelerator_collective(backend_name, tensor, op_name=op_name, **kwargs)
    if hw_res is not None:
        return hw_res

    if raw_inst is not None and isinstance(raw_inst, _CollectiveBackendProto):
        backend_res = raw_inst.execute_collective(op_name, tensor, **kwargs)
        if backend_res is not None:
            return backend_res

    return _hc_mod._dispatch_host_collective(op_name, tensor, **kwargs)
