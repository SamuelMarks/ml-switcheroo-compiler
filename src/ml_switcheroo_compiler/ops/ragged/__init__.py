# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Ragged tensor operations for handling variable-length sequential data.

This module provides operations for working with ragged tensors, which are
tensors with non-uniform shapes across one or more dimensions.
"""

from typing import Any

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .core import RaggedDot


@register_op("RaggedGather")
class RaggedGather(OpDef):
    """Operation definition for gathering elements into a ragged tensor.

    This operation gathers elements from the `params` tensor according to the
    `indices` tensor, resulting in a ragged tensor structure.
    """

    op_name = "RaggedGather"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedTensorToDense")
class RaggedTensorToDense(OpDef):
    """Operation definition for converting a ragged tensor to a dense tensor.

    This operation transforms a ragged tensor into a dense tensor by padding
    the variable-length dimensions with a default value.
    """

    op_name = "RaggedTensorToDense"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape when converting to a dense tensor.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


def ragged_tensor_to_dense(
    rt_input: "Tensor",  # type: ignore
    default_value: Any = None,
    row_partition_tensors: Any = None,
    row_partition_types: Any = None,
    shape: Any = None,
) -> "Tensor":  # type: ignore
    """Convert a ragged tensor representation into a regular dense tensor.

    Pads the variable-length dimensions of the input ragged tensor with the
    specified default value to create a dense tensor with uniform dimensions.

    Args:
        rt_input: The input ragged tensor to convert.
        default_value: The value used to pad the ragged dimensions.
        row_partition_tensors: Tensors defining the row partitions.
        row_partition_types: String types of the row partitions.
        shape: The target shape of the output dense tensor.

    Returns:
        A dense tensor padded to have uniform dimensions.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "RaggedTensorToDense",
            rt_input.data,
            default_value=default_value,
            row_partition_tensors=row_partition_tensors,
            row_partition_types=row_partition_types,
            shape=shape,
        )
        return Tensor(data, rt_input.config)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(  # type: ignore
        "RaggedTensorToDense",
        [rt_input],
        {
            "default_value": default_value,
            "row_partition_tensors": row_partition_tensors,
            "row_partition_types": row_partition_types,
            "shape": shape,
        },
        (),
        rt_input.dtype,
    )


@register_op("RaggedAdd")
class RaggedAdd(OpDef):
    """Operation definition for adding two ragged tensors element-wise.

    This operation computes the element-wise sum of two ragged tensors that
    have compatible shapes.
    """

    op_name = "RaggedAdd"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged addition operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedMatMul")
class RaggedMatMul(OpDef):
    """Operation definition for ragged matrix multiplication.

    This operation performs matrix multiplication where one or both of the
    inputs may be a ragged tensor.
    """

    op_name = "RaggedMatMul"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged matrix multiplication.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedDynamicBroadcast")
class RaggedDynamicBroadcast(OpDef):
    """Operation definition for dynamically broadcasting a ragged tensor.

    This operation broadcasts a ragged tensor to a new shape based on dynamic
    dimension information.
    """

    op_name = "RaggedDynamicBroadcast"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape after a ragged dynamic broadcast.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedConstant")
class RaggedConstant(OpDef):
    """Operation definition for creating a ragged tensor from a constant.

    This operation constructs a ragged tensor using values provided in a
    nested python list or similar constant structure.
    """

    op_name = "RaggedConstant"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged constant operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedCrossHashed")
class RaggedCrossHashed(OpDef):
    """Operation definition for a hashed ragged cross product.

    This operation computes a cross product of ragged tensor elements and
    hashes the results for efficiency.
    """

    op_name = "RaggedCrossHashed"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged cross hashed operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedRange")
class RaggedRange(OpDef):
    """Operation definition for creating a ragged sequence of numbers.

    This operation generates a ragged tensor containing sequences of numbers
    based on specified starts, limits, and deltas.
    """

    op_name = "RaggedRange"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged range operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedRowSplitsToSegmentIds")
class RaggedRowSplitsToSegmentIds(OpDef):
    """Operation definition for converting row splits to segment IDs.

    This operation takes a row splits tensor defining ragged row boundaries
    and converts it into an equivalent segment IDs tensor.
    """

    op_name = "RaggedRowSplitsToSegmentIds"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape when converting row splits to segment IDs.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedSegmentIdsToRowSplits")
class RaggedSegmentIdsToRowSplits(OpDef):
    """Operation definition for converting segment IDs to row splits.

    This operation takes a segment IDs tensor defining ragged elements
    and converts it into an equivalent row splits tensor.
    """

    op_name = "RaggedSegmentIdsToRowSplits"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape when converting segment IDs to row splits.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedStack")
class RaggedStack(OpDef):
    """Operation definition for stacking ragged tensors.

    This operation stacks a list of ragged tensors along a specified axis
    to create a higher-rank ragged tensor.
    """

    op_name = "RaggedStack"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged stack operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("RaggedStackDynamicPartitions")
class RaggedStackDynamicPartitions(OpDef):
    """Operation definition for dynamically partitioning and stacking.

    This operation partitions the elements of a ragged tensor and then stacks
    them dynamically based on the given partitions.
    """

    op_name = "RaggedStackDynamicPartitions"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the output shape for a ragged stack dynamic partitions op.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


__all__ = [
    "RaggedGather",
    "RaggedTensorToDense",
    "ragged_tensor_to_dense",
    "RaggedAdd",
    "RaggedMatMul",
    "RaggedDynamicBroadcast",
    "RaggedConstant",
    "RaggedCrossHashed",
    "RaggedRange",
    "RaggedRowSplitsToSegmentIds",
    "RaggedSegmentIdsToRowSplits",
    "RaggedStack",
    "RaggedStackDynamicPartitions",
    "RaggedDot",
    "BooleanMask",
    "MapFlatValues",
    "boolean_mask",
    "map_flat_values",
]


@register_op("BooleanMask")
class BooleanMask(OpDef):
    """Boolean mask operation."""

    op_name = "BooleanMask"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("MapFlatValues")
class MapFlatValues(OpDef):
    """Map flat values operation."""

    op_name = "MapFlatValues"

    def infer_shape(self, op: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            op (object): The op parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


from .frontend import boolean_mask, map_flat_values
