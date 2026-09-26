# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

    def infer_shape(self, *args, **kwargs):
        """Infer output shape for RaggedGather.

        Args:
            *args (object): Positional args (params, indices).
            **kwargs (object): Keyword args.

        Returns:
            tuple: Inferred shape.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        params = args[0] if args else kwargs.get("params")
        indices = args[1] if len(args) > 1 else kwargs.get("indices")
        if params is None or indices is None:
            return ()
        shape_p = tuple(getattr(params, "shape", getattr(params, "shape_metadata", params if isinstance(params, (list, tuple)) else ())))
        shape_i = tuple(getattr(indices, "shape", getattr(indices, "shape_metadata", indices if isinstance(indices, (list, tuple)) else ())))
        leading = shape_i if shape_i else (SymVar("gather_batch"),)
        trailing = shape_p[1:] if len(shape_p) > 1 else ()
        return leading + trailing


@register_op("RaggedTensorToDense")
class RaggedTensorToDense(OpDef):
    """Operation definition for converting a ragged tensor to a dense tensor.

    This operation transforms a ragged tensor into a dense tensor by padding
    the variable-length dimensions with a default value.
    """

    op_name = "RaggedTensorToDense"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape when converting to a dense tensor.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Dense output shape.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        target_shape = kwargs.get("shape")
        if target_shape is not None:
            return tuple(target_shape)
        rt_input = args[0] if args else kwargs.get("rt_input")
        if rt_input is None:
            return ()
        inp_shape = tuple(getattr(rt_input, "shape", getattr(rt_input, "shape_metadata", rt_input if isinstance(rt_input, (list, tuple)) else ())))
        if not inp_shape:
            return ()
        batch = inp_shape[0]
        ragged_dim = inp_shape[1] if len(inp_shape) > 1 else SymVar("max_seq_len")
        trailing = inp_shape[2:] if len(inp_shape) > 2 else ()
        return (batch, ragged_dim) + trailing


def ragged_tensor_to_dense(
    rt_input: "Tensor",
    default_value=None,
    row_partition_tensors=None,
    row_partition_types=None,
    shape=None,
) -> "Tensor":
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

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged addition operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result shape.
        """
        inp = args[0] if args else kwargs.get("x", kwargs.get("a"))
        if inp is None:
            return ()
        return tuple(getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))


@register_op("RaggedMatMul")
class RaggedMatMul(OpDef):
    """Operation definition for ragged matrix multiplication.

    This operation performs matrix multiplication where one or both of the
    inputs may be a ragged tensor.
    """

    op_name = "RaggedMatMul"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged matrix multiplication.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Output shape with preserved outer and contraction dims.
        """
        a = args[0] if args else kwargs.get("a")
        b = args[1] if len(args) > 1 else kwargs.get("b")
        if a is None or b is None:
            return ()
        shape_a = tuple(getattr(a, "shape", getattr(a, "shape_metadata", a if isinstance(a, (list, tuple)) else ())))
        shape_b = tuple(getattr(b, "shape", getattr(b, "shape_metadata", b if isinstance(b, (list, tuple)) else ())))
        if len(shape_a) < 2 or len(shape_b) < 2:
            return shape_a
        m = shape_a[-2]
        n = shape_b[-1]
        batch = shape_a[:-2]
        return batch + (m, n)


@register_op("RaggedDynamicBroadcast")
class RaggedDynamicBroadcast(OpDef):
    """Operation definition for dynamically broadcasting a ragged tensor.

    This operation broadcasts a ragged tensor to a new shape based on dynamic
    dimension information.
    """

    op_name = "RaggedDynamicBroadcast"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape after a ragged dynamic broadcast.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Broadcasted target shape.
        """
        shape = kwargs.get("shape", args[1] if len(args) > 1 else None)
        if shape is not None:
            return tuple(shape)
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        return tuple(getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))


@register_op("RaggedConstant")
class RaggedConstant(OpDef):
    """Operation definition for creating a ragged tensor from a constant.

    This operation constructs a ragged tensor using values provided in a
    nested python list or similar constant structure.
    """

    op_name = "RaggedConstant"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged constant operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result shape with symbolic inner ragged dimension.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        val = args[0] if args else kwargs.get("pylist", kwargs.get("constant", kwargs.get("value")))
        if val is None:
            return ()
        if isinstance(val, (list, tuple)):
            batch = len(val)
            return (batch, SymVar("var_len"))
        return ()


@register_op("RaggedCrossHashed")
class RaggedCrossHashed(OpDef):
    """Operation definition for a hashed ragged cross product.

    This operation computes a cross product of ragged tensor elements and
    hashes the results for efficiency.
    """

    op_name = "RaggedCrossHashed"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged cross hashed operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result shape with leading batch and symbolic cross length.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        inputs = args[0] if (args and isinstance(args[0], (list, tuple))) else list(args)
        if not inputs and "inputs" in kwargs:
            inputs = kwargs["inputs"]
        if not inputs:
            return ()
        first = inputs[0]
        f_shape = tuple(getattr(first, "shape", getattr(first, "shape_metadata", first if isinstance(first, (list, tuple)) else ())))
        batch = f_shape[0] if f_shape else 1
        return (batch, SymVar("cross_len"))


@register_op("RaggedRange")
class RaggedRange(OpDef):
    """Operation definition for creating a ragged sequence of numbers.

    This operation generates a ragged tensor containing sequences of numbers
    based on specified starts, limits, and deltas.
    """

    op_name = "RaggedRange"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged range operation.

        Args:
            *args (object): Positional args (starts, limits, deltas).
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result shape with batch size and symbolic range length.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        starts = args[0] if args else kwargs.get("starts")
        if starts is None:
            return (1, SymVar("range_len"))
        s_shape = tuple(getattr(starts, "shape", getattr(starts, "shape_metadata", (len(starts),) if isinstance(starts, (list, tuple)) else ())))
        batch = s_shape[0] if s_shape else 1
        return (batch, SymVar("range_len"))


@register_op("RaggedRowSplitsToSegmentIds")
class RaggedRowSplitsToSegmentIds(OpDef):
    """Operation definition for converting row splits to segment IDs.

    This operation takes a row splits tensor defining ragged row boundaries
    and converts it into an equivalent segment IDs tensor.
    """

    op_name = "RaggedRowSplitsToSegmentIds"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape when converting row splits to segment IDs.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: 1D segment IDs shape.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        return (SymVar("num_elements"),)


@register_op("RaggedSegmentIdsToRowSplits")
class RaggedSegmentIdsToRowSplits(OpDef):
    """Operation definition for converting segment IDs to row splits.

    This operation takes a segment IDs tensor defining ragged elements
    and converts it into an equivalent row splits tensor.
    """

    op_name = "RaggedSegmentIdsToRowSplits"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape when converting segment IDs to row splits.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Row splits shape (num_segments + 1,).
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        num_segments = kwargs.get("num_segments")
        if isinstance(num_segments, int):
            return (num_segments + 1,)
        return (SymVar("num_segments_plus_1"),)


@register_op("RaggedStack")
class RaggedStack(OpDef):
    """Operation definition for stacking ragged tensors.

    This operation stacks a list of ragged tensors along a specified axis
    to create a higher-rank ragged tensor.
    """

    op_name = "RaggedStack"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged stack operation.

        Args:
            *args (object): Positional args (list of tensors).
            **kwargs (object): Keyword args.

        Returns:
            tuple: Higher-rank stacked shape.
        """
        rt_inputs = args[0] if (args and isinstance(args[0], (list, tuple))) else list(args)
        if not rt_inputs and "values" in kwargs:
            rt_inputs = kwargs["values"]
        if not rt_inputs:
            return ()
        first = rt_inputs[0]
        f_shape = tuple(getattr(first, "shape", getattr(first, "shape_metadata", first if isinstance(first, (list, tuple)) else ())))
        axis = int(kwargs.get("axis", 0))
        n = len(rt_inputs)
        s = list(f_shape)
        if axis < 0:
            axis += len(s) + 1
        s.insert(axis, n)
        return tuple(s)


@register_op("RaggedStackDynamicPartitions")
class RaggedStackDynamicPartitions(OpDef):
    """Operation definition for dynamically partitioning and stacking.

    This operation partitions the elements of a ragged tensor and then stacks
    them dynamically based on the given partitions.
    """

    op_name = "RaggedStackDynamicPartitions"

    def infer_shape(self, *args, **kwargs):
        """Calculate the output shape for a ragged stack dynamic partitions op.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Partition stacked shape (num_partitions, ...).
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        num_partitions = int(kwargs.get("num_partitions", 2))
        data = args[0] if args else kwargs.get("data")
        trailing = ()
        if data is not None:
            d_shape = tuple(getattr(data, "shape", getattr(data, "shape_metadata", data if isinstance(data, (list, tuple)) else ())))
            trailing = d_shape[1:] if len(d_shape) > 1 else ()
        return (num_partitions, SymVar("partition_len")) + trailing


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

    def infer_shape(self, *args, **kwargs):
        """Infer output shape for BooleanMask.

        Args:
            *args (object): Positional args (data, mask).
            **kwargs (object): Keyword args.

        Returns:
            tuple: Filtered shape with dynamic leading dimension.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        data = args[0] if args else kwargs.get("data", kwargs.get("tensor"))
        mask = args[1] if len(args) > 1 else kwargs.get("mask")
        if data is None:
            return (SymVar("masked_count"),)
        d_shape = tuple(getattr(data, "shape", getattr(data, "shape_metadata", data if isinstance(data, (list, tuple)) else ())))
        m_ndim = 1
        if mask is not None:
            m_shape = getattr(mask, "shape", getattr(mask, "shape_metadata", None))
            if m_shape:
                m_ndim = len(m_shape)
        trailing = d_shape[m_ndim:] if len(d_shape) >= m_ndim else ()
        return (SymVar("masked_count"),) + trailing


@register_op("MapFlatValues")
class MapFlatValues(OpDef):
    """Map flat values operation."""

    op_name = "MapFlatValues"

    def infer_shape(self, op=None, *args, **kwargs):
        """Infer shape for MapFlatValues operation.

        Args:
            op (object): The mapped op.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Shape matching the primary ragged input.
        """
        inp = args[0] if args else kwargs.get("x", kwargs.get("input"))
        if inp is None:
            return ()
        return tuple(getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))


from .frontend import boolean_mask, map_flat_values
