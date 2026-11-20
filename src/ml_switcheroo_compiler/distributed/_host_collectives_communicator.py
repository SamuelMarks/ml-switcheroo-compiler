"""In-memory multi-rank host collective communicator and eager reduction kernels."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol
from ml_switcheroo_compiler.core.shape import (
    IRAttrValue,
    OpValue,
    SliceProtocol,
)
from ml_switcheroo_compiler.distributed._host_collectives_protocols import (
    TensorLike,
    _copy_if_possible,
    _hc_extract_shape,
)


def _exec_collective_op(op_name: str, *args: OpValue, **kwargs: IRAttrValue) -> OpValue:
    """Execute an elementary operation via the reference backend.

    Args:
        op_name (str): Canonical IR operation name ('Add', 'Multiply', 'Maximum', 'Minimum', 'Concatenate', 'Divide').
        *args (OpValue): Positional tensor operands.
        **kwargs (IRAttrValue): Keyword operation attributes.

    Returns:
        OpValue: Evaluated tensor buffer.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    return get_active_backend().execute_op(op_name, *args, **kwargs)


@typing.runtime_checkable
class _SliceableBufferProtocol(typing.Protocol):
    """Protocol for tensor buffers supporting multi-dimensional slice indexing."""

    @property
    def ndim(self) -> int:
        """Return number of tensor dimensions.

        Returns:
            int: Tensor rank.
        """
        ...

    def __getitem__(self, key: tuple[SliceProtocol, ...]) -> TensorLike:
        """Index tensor buffer with a tuple of slices.

        Args:
            key (tuple[SliceProtocol, ...]): Per-dimension slice tuple.

        Returns:
            TensorLike: Sliced tensor buffer.
        """
        ...


def _as_sliceable_buffer(val: TensorLike | _SliceableBufferProtocol) -> _SliceableBufferProtocol:
    """Verify and narrow ``val`` to ``_SliceableBufferProtocol``.

    Args:
        val (TensorLike | _SliceableBufferProtocol): Candidate tensor buffer.

    Returns:
        _SliceableBufferProtocol: Verified sliceable buffer.
    """
    assert isinstance(val, _SliceableBufferProtocol) and not isinstance(val, ModuleType)
    return val


def _split_along_axis(data: TensorLike, num_splits: int, axis: int = 0) -> list[TensorLike]:
    """Split a tensor along the specified axis into num_splits partitions.

    Args:
        data (TensorLike): Input tensor buffer.
        num_splits (int): Number of partitions to produce.
        axis (int): Axis along which to partition.

    Returns:
        list[TensorLike]: List of partitioned tensor slices.
    """
    raw_data: OpValue = data
    shape = _hc_extract_shape(raw_data.shape) if isinstance(raw_data, ShapeProtocol) else ()
    ndim = len(shape)
    norm_axis = axis if axis >= 0 else ndim + axis
    axis_len = shape[norm_axis]
    base_size, remainder = divmod(axis_len, max(1, num_splits))
    sliceable = _as_sliceable_buffer(data)
    slices: list[TensorLike] = []
    start = 0
    for idx in range(max(1, num_splits)):
        end = start + base_size + (1 if idx < remainder else 0)
        key: tuple[SliceProtocol, ...] = tuple(
            slice(start, end) if d == norm_axis else slice(None) for d in range(ndim)
        )
        slices.append(sliceable[key])
        start = end
    return slices


def _all_reduce_eager(
    data: TensorLike,
    op: str = "SUM",
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-reduce across rank tensors using backend operations.

    Args:
        data (TensorLike): Local rank tensor data.
        op (str): Reduction operator ('SUM', 'PROD', 'MAX', 'MIN').
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Reduced tensor buffer.
    """
    if all_ranks_data is None:
        return _copy_if_possible(data)

    items = list(all_ranks_data)
    acc: OpValue = _copy_if_possible(items[0])
    norm_op: str = op.upper()
    if norm_op == "SUM":
        ir_op = "Add"
    elif norm_op == "MAX":
        ir_op = "Maximum"
    elif norm_op == "MIN":
        ir_op = "Minimum"
    elif norm_op == "PROD":
        ir_op = "Multiply"
    else:
        ir_op = "Add"
    for item in items[1:]:
        acc = _exec_collective_op(ir_op, acc, item)
    return acc


def _all_gather_eager(
    data: TensorLike,
    axis: int = 0,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-gather across rank tensors using backend Concatenate.

    Args:
        data (TensorLike): Local rank tensor data.
        axis (int): Axis along which to concatenate rank tensors.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Gathered tensor buffer.
    """
    if all_ranks_data is None:
        return _copy_if_possible(data)
    return _exec_collective_op("Concatenate", list(all_ranks_data), axis=axis)


def _reduce_scatter_eager(
    data: TensorLike,
    op: str = "SUM",
    scatter_dim: int = 0,
    rank: int = 0,
    world_size: int = 1,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager reduce-scatter across rank tensors.

    Args:
        data (TensorLike): Local rank tensor data.
        op (str): Reduction operator ('SUM', 'PROD', 'MAX', 'MIN').
        scatter_dim (int): Dimension along which to scatter reduced slices.
        rank (int): Current process rank index.
        world_size (int): Total number of participating ranks.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Reduced and scattered tensor slice for the current rank.
    """
    reduced: TensorLike = _all_reduce_eager(data, op=op, all_ranks_data=all_ranks_data)
    chunks: list[TensorLike] = _split_along_axis(reduced, world_size, axis=scatter_dim)
    return chunks[rank]


def _all_to_all_eager(
    data: TensorLike,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    rank: int = 0,
    world_size: int = 1,
    all_ranks_data: Sequence[TensorLike] | None = None,
) -> TensorLike:
    """Perform an eager all-to-all exchange across rank tensors.

    Args:
        data (TensorLike): Local rank tensor data.
        scatter_dim (int): Dimension along which to split outgoing chunks.
        gather_dim (int): Dimension along which to concatenate incoming chunks.
        rank (int): Current process rank index.
        world_size (int): Total number of participating ranks.
        all_ranks_data (Sequence[TensorLike] | None): Optional sequence of tensors from all ranks.

    Returns:
        TensorLike: Exchanged tensor buffer for the current rank.
    """
    if all_ranks_data is None:
        return _copy_if_possible(data)
    split_ranks: list[list[TensorLike]] = [
        _split_along_axis(rank_arr, world_size, axis=scatter_dim) for rank_arr in all_ranks_data
    ]
    rank_slices: list[TensorLike] = [split_ranks[r][rank] for r in range(world_size)]
    return _exec_collective_op("Concatenate", rank_slices, axis=gather_dim)


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
        data: TensorLike,
        op: str = "SUM",
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Perform an all-reduce operation across ranks.

        Args:
            data (TensorLike): Tensor payload for current rank.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Reduced output tensor.
        """
        return _all_reduce_eager(data, op=op, all_ranks_data=all_ranks_data)

    def all_gather(
        self,
        data: TensorLike,
        axis: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Gather tensors from all ranks along a specified axis.

        Args:
            data (TensorLike): Tensor payload for current rank.
            axis (int): Axis along which to concatenate gathered tensors.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Concatenated gathered output tensor.
        """
        return _all_gather_eager(data, axis=axis, all_ranks_data=all_ranks_data)

    def reduce_scatter(
        self,
        data: TensorLike,
        op: str = "SUM",
        scatter_dim: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Reduce tensors across ranks then scatter segments to individual ranks.

        Args:
            data (TensorLike): Input tensor.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            scatter_dim (int): Dimension along which to partition and scatter.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Scattering chunk for the current rank.
        """
        return _reduce_scatter_eager(
            data,
            op=op,
            scatter_dim=scatter_dim,
            rank=self.rank,
            world_size=self.world_size,
            all_ranks_data=all_ranks_data,
        )

    def all_to_all(
        self,
        data: TensorLike,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Perform all-to-all communication among ranks.

        Args:
            data (TensorLike): Input tensor for current rank.
            scatter_dim (int): Axis along which inputs are split.
            gather_dim (int): Axis along which collected outputs are concatenated.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Evaluated all-to-all output tensor.
        """
        return _all_to_all_eager(
            data,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            rank=self.rank,
            world_size=self.world_size,
            all_ranks_data=all_ranks_data,
        )

    def broadcast(
        self,
        data: TensorLike,
        root: int = 0,
        all_ranks_data: Sequence[TensorLike] | None = None,
    ) -> TensorLike:
        """Broadcast tensor from root rank to all ranks.

        Args:
            data (TensorLike): Input tensor for current rank.
            root (int): Rank originating the broadcast.
            all_ranks_data (Sequence[TensorLike] | None): Sequence of inputs from all ranks.

        Returns:
            TensorLike: Broadcasted tensor matching root rank payload.
        """
        if all_ranks_data is None:
            return _copy_if_possible(data)
        src = all_ranks_data[root]
        return _copy_if_possible(src)
