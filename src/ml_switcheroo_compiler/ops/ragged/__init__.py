# ruff: noqa: E402, D101, D102
"""Ragged tensor operations for handling variable-length sequential data.

This module provides operations for working with ragged tensors, which are
tensors with non-uniform shapes across one or more dimensions.
"""

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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged gather operation.

        Args:
            *args: Positional arguments.
            params: The source tensor from which to gather values.
            indices: The index tensor specifying which elements to gather.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the ragged gather operation result.
        """
        return ()


@register_op("RaggedTensorToDense")
class RaggedTensorToDense(OpDef):
    """Operation definition for converting a ragged tensor to a dense tensor.

    This operation transforms a ragged tensor into a dense tensor by padding
    the variable-length dimensions with a default value.
    """

    op_name = "RaggedTensorToDense"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape when converting to a dense tensor.

        Args:
            *args: Positional arguments.
            rt_input: The input ragged tensor to convert.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the resulting dense tensor.
        """
        return ()


def ragged_tensor_to_dense(
    rt_input: "Tensor",
    default_value: object = None,
    row_partition_tensors: object = None,
    row_partition_types: object = None,
    shape: object = None,
) -> "Tensor":
    """Converts a ragged tensor representation into a regular dense tensor.

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

    return _emit_shape_node(
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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged addition operation.

        Args:
            *args: Positional arguments.
            a: The first input ragged tensor.
            b: The second input ragged tensor.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the ragged addition result.
        """
        return ()


@register_op("RaggedMatMul")
class RaggedMatMul(OpDef):
    """Operation definition for ragged matrix multiplication.

    This operation performs matrix multiplication where one or both of the
    inputs may be a ragged tensor.
    """

    op_name = "RaggedMatMul"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged matrix multiplication.

        Args:
            *args: Positional arguments.
            a: The first input tensor (potentially ragged).
            b: The second input tensor (potentially ragged).
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the ragged matrix multiplication result.
        """
        return ()


@register_op("RaggedDynamicBroadcast")
class RaggedDynamicBroadcast(OpDef):
    """Operation definition for dynamically broadcasting a ragged tensor.

    This operation broadcasts a ragged tensor to a new shape based on dynamic
    dimension information.
    """

    op_name = "RaggedDynamicBroadcast"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape after a ragged dynamic broadcast.

        Args:
            *args: Positional arguments.
            rt_input: The input ragged tensor to broadcast.
            shape: The target shape to broadcast to.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the broadcasted ragged tensor.
        """
        return ()


@register_op("RaggedConstant")
class RaggedConstant(OpDef):
    """Operation definition for creating a ragged tensor from a constant.

    This operation constructs a ragged tensor using values provided in a
    nested python list or similar constant structure.
    """

    op_name = "RaggedConstant"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged constant operation.

        Args:
            *args: Positional arguments.
            pylist: The nested python list representing the ragged constant.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the created ragged constant tensor.
        """
        return ()


@register_op("RaggedCrossHashed")
class RaggedCrossHashed(OpDef):
    """Operation definition for a hashed ragged cross product.

    This operation computes a cross product of ragged tensor elements and
    hashes the results for efficiency.
    """

    op_name = "RaggedCrossHashed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged cross hashed operation.

        Args:
            *args: Positional arguments.
            inputs: A list or tuple of ragged tensors to cross.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the hashed cross product result.
        """
        return ()


@register_op("RaggedRange")
class RaggedRange(OpDef):
    """Operation definition for creating a ragged sequence of numbers.

    This operation generates a ragged tensor containing sequences of numbers
    based on specified starts, limits, and deltas.
    """

    op_name = "RaggedRange"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged range operation.

        Args:
            *args: Positional arguments.
            starts: The starting values for the sequences.
            limits: The upper limits (exclusive) for the sequences.
            deltas: The step sizes for the sequences.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the generated ragged range tensor.
        """
        return ()


@register_op("RaggedRowSplitsToSegmentIds")
class RaggedRowSplitsToSegmentIds(OpDef):
    """Operation definition for converting row splits to segment IDs.

    This operation takes a row splits tensor defining ragged row boundaries
    and converts it into an equivalent segment IDs tensor.
    """

    op_name = "RaggedRowSplitsToSegmentIds"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape when converting row splits to segment IDs.

        Args:
            *args: Positional arguments.
            splits: The row splits tensor defining the ragged boundaries.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the resulting segment IDs tensor.
        """
        return ()


@register_op("RaggedSegmentIdsToRowSplits")
class RaggedSegmentIdsToRowSplits(OpDef):
    """Operation definition for converting segment IDs to row splits.

    This operation takes a segment IDs tensor defining ragged elements
    and converts it into an equivalent row splits tensor.
    """

    op_name = "RaggedSegmentIdsToRowSplits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape when converting segment IDs to row splits.

        Args:
            *args: Positional arguments.
            segment_ids: The segment IDs tensor to convert.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the resulting row splits tensor.
        """
        return ()


@register_op("RaggedStack")
class RaggedStack(OpDef):
    """Operation definition for stacking ragged tensors.

    This operation stacks a list of ragged tensors along a specified axis
    to create a higher-rank ragged tensor.
    """

    op_name = "RaggedStack"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged stack operation.

        Args:
            *args: Positional arguments.
            values: A list or tuple of ragged tensors to stack.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the stacked ragged tensor.
        """
        return ()


@register_op("RaggedStackDynamicPartitions")
class RaggedStackDynamicPartitions(OpDef):
    """Operation definition for dynamically partitioning and stacking.

    This operation partitions the elements of a ragged tensor and then stacks
    them dynamically based on the given partitions.
    """

    op_name = "RaggedStackDynamicPartitions"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Calculates the output shape for a ragged stack dynamic partitions op.

        Args:
            *args: Positional arguments.
            data: The input tensor containing the data to partition.
            partitions: The tensor specifying which partition each element goes to.
            num_partitions: The total number of partitions.
            **kwargs: Additional keyword arguments for the shape inference.

        Returns:
            The inferred shape tuple for the dynamically partitioned ragged tensor.
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
    op_name = "BooleanMask"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("MapFlatValues")
class MapFlatValues(OpDef):
    op_name = "MapFlatValues"

    def infer_shape(self, op: object, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


from .frontend import boolean_mask, map_flat_values
