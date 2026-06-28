"""Ragged ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from .frontend import (
    ragged_constant as ragged_constant,
    ragged_cross as ragged_cross,
    ragged_cross_hashed as ragged_cross_hashed,
    ragged_range as ragged_range,
    ragged_row_splits_to_segment_ids as ragged_row_splits_to_segment_ids,
    ragged_segment_ids_to_row_splits as ragged_segment_ids_to_row_splits,
    ragged_stack as ragged_stack,
    ragged_stack_dynamic_partitions as ragged_stack_dynamic_partitions,
    ragged_dot as ragged_dot,
)

from .core import RaggedDot


@register_op("RaggedGather")
class RaggedGather(OpDef):
    """RaggedGather operation."""

    op_name = "RaggedGather"

    def infer_shape(self, params: object, indices: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedTensorToDense")
class RaggedTensorToDense(OpDef):
    """RaggedTensorToDense operation."""

    op_name = "RaggedTensorToDense"

    def infer_shape(self, rt_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedAdd")
class RaggedAdd(OpDef):
    """RaggedAdd operation."""

    op_name = "RaggedAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedMatMul")
class RaggedMatMul(OpDef):
    """RaggedMatMul operation."""

    op_name = "RaggedMatMul"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedDynamicBroadcast")
class RaggedDynamicBroadcast(OpDef):
    """RaggedDynamicBroadcast operation."""

    op_name = "RaggedDynamicBroadcast"

    def infer_shape(self, rt_input: object, shape: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedConstant")
class RaggedConstant(OpDef):
    """RaggedConstant operation."""

    op_name = "RaggedConstant"

    def infer_shape(self, pylist: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover  # pragma: no cover


@register_op("RaggedCrossHashed")
class RaggedCrossHashed(OpDef):
    """RaggedCrossHashed operation."""

    op_name = "RaggedCrossHashed"

    def infer_shape(self, inputs: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedRange")
class RaggedRange(OpDef):
    """RaggedRange operation."""

    op_name = "RaggedRange"

    def infer_shape(
        self, starts: object, limits: object, deltas: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


@register_op("RaggedRowSplitsToSegmentIds")
class RaggedRowSplitsToSegmentIds(OpDef):
    """RaggedRowSplitsToSegmentIds operation."""

    op_name = "RaggedRowSplitsToSegmentIds"

    def infer_shape(self, splits: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedSegmentIdsToRowSplits")
class RaggedSegmentIdsToRowSplits(OpDef):
    """RaggedSegmentIdsToRowSplits operation."""

    op_name = "RaggedSegmentIdsToRowSplits"

    def infer_shape(self, segment_ids: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedStack")
class RaggedStack(OpDef):
    """RaggedStack operation."""

    op_name = "RaggedStack"

    def infer_shape(self, values: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("RaggedStackDynamicPartitions")
class RaggedStackDynamicPartitions(OpDef):
    """RaggedStackDynamicPartitions operation."""

    op_name = "RaggedStackDynamicPartitions"

    def infer_shape(
        self, data: object, partitions: object, num_partitions: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


__all__ = [
    "RaggedAdd",
    "RaggedConstant",
    "RaggedCrossHashed",
    "RaggedDot",
    "RaggedDynamicBroadcast",
    "RaggedGather",
    "RaggedMatMul",
    "RaggedRange",
    "RaggedRowSplitsToSegmentIds",
    "RaggedSegmentIdsToRowSplits",
    "RaggedStack",
    "RaggedStackDynamicPartitions",
    "RaggedTensorToDense",
    "ragged_constant",
    "ragged_cross",
    "ragged_cross_hashed",
    "ragged_dot",
    "ragged_range",
    "ragged_row_splits_to_segment_ids",
    "ragged_segment_ids_to_row_splits",
    "ragged_stack",
    "ragged_stack_dynamic_partitions",
]
