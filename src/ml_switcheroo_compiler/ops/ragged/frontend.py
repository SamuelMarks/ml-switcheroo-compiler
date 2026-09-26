"""Ragged frontend operations."""

from __future__ import annotations

from typing import Union

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

RaggedArg = Union[Tensor, int, float, str, bool, list[int], tuple[int, ...], None]
RaggedKwarg = Union[int, float, str, bool, tuple[int, ...], None]


def _ragged_op(
    op_name: str,
    *args: RaggedArg,
    **kwargs: RaggedKwarg,
) -> Tensor | tuple[Tensor, ...]:
    """Dispatch a ragged operation either eagerly or via IR node emission.

    Args:
        op_name (str): The name of the registered ragged operation.
        *args (RaggedArg): Positional tensor or scalar arguments.
        **kwargs (RaggedKwarg): Keyword configuration arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: The evaluated or symbolic result.
    """
    op_cls = get_op(op_name)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(op_name, *[getattr(a, "data", a) for a in args], **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                getattr(args[0], "dtype", "float32") if args else "float32",
                getattr(args[0], "device", "cpu") if args else "cpu",
            ),
        )

    op = op_cls()
    out_shape = op.infer_shape(*args, **kwargs)
    tensor_inputs = [a for a in args if isinstance(a, Tensor)]
    return _emit_linalg_node(
        op_name,
        tensor_inputs,
        kwargs,
        [tuple(out_shape)],
        [getattr(args[0], "dtype", "float32") if args else "float32"],
    )


def ragged_constant(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Construct a ragged constant tensor.

    Args:
        *args (RaggedArg): Constant values or nested lists.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Ragged constant tensor.
    """
    return _ragged_op("RaggedConstant", *args, **kwargs)


def ragged_cross(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Compute cross product of ragged tensors.

    Args:
        *args (RaggedArg): Input ragged tensors.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Crossed ragged tensor.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_cross_hashed(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Compute hashed feature crosses of ragged tensors.

    Args:
        *args (RaggedArg): Input ragged tensors.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Hashed crossed ragged tensor.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_range(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Generate ragged range sequences.

    Args:
        *args (RaggedArg): Start, limit, and delta arguments.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Ragged range tensor.
    """
    return _ragged_op("RaggedRange", *args, **kwargs)


def ragged_row_splits_to_segment_ids(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Convert row splits representation to segment IDs.

    Args:
        *args (RaggedArg): Row splits tensor.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Segment IDs tensor.
    """
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)


def ragged_segment_ids_to_row_splits(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Convert segment IDs representation to row splits.

    Args:
        *args (RaggedArg): Segment IDs tensor.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Row splits tensor.
    """
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)


def ragged_stack(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Stack a list of rank-R tensors into a rank-(R+1) ragged tensor.

    Args:
        *args (RaggedArg): Tensors to stack.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Stacked ragged tensor.
    """
    return _ragged_op("RaggedStack", *args, **kwargs)


def ragged_stack_dynamic_partitions(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Stack dynamic partitions into a ragged tensor.

    Args:
        *args (RaggedArg): Partitions to stack.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Stacked partitions ragged tensor.
    """
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)


def ragged_gather(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Gather slices from params according to ragged indices.

    Args:
        *args (RaggedArg): Params and indices tensors.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Gathered ragged tensor.
    """
    return _ragged_op("RaggedGather", *args, **kwargs)


def ragged_tensor_to_dense(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Convert a ragged tensor into a dense tensor.

    Args:
        *args (RaggedArg): Input ragged tensor.
        **kwargs (RaggedKwarg): Additional keyword arguments such as default_value or shape.

    Returns:
        Tensor | tuple[Tensor, ...]: Dense tensor representation.
    """
    return _ragged_op("RaggedTensorToDense", *args, **kwargs)


def ragged_add(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Add two ragged tensors elementwise.

    Args:
        *args (RaggedArg): Ragged tensors x and y.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Sum ragged tensor.
    """
    return _ragged_op("RaggedAdd", *args, **kwargs)


def ragged_matmul(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Perform matrix multiplication of ragged tensors.

    Args:
        *args (RaggedArg): Matrices a and b.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Resulting matrix product.
    """
    return _ragged_op("RaggedMatMul", *args, **kwargs)


def ragged_dynamic_broadcast(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Broadcast ragged tensor dynamically.

    Args:
        *args (RaggedArg): Input ragged tensor.
        **kwargs (RaggedKwarg): Target broadcast shape keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Broadcasted ragged tensor.
    """
    return _ragged_op("RaggedDynamicBroadcast", *args, **kwargs)


def ragged_dot(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Compute dot product of ragged tensors.

    Args:
        *args (RaggedArg): Tensors a and b.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Dot product tensor.
    """
    return _ragged_op("RaggedDot", *args, **kwargs)


def boolean_mask(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Apply boolean mask to a tensor producing ragged or masked output.

    Args:
        *args (RaggedArg): Input tensor and boolean mask tensor.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Masked output tensor.
    """
    return _ragged_op("BooleanMask", *args, **kwargs)


def map_flat_values(*args: RaggedArg, **kwargs: RaggedKwarg) -> Tensor | tuple[Tensor, ...]:
    """Map a callable or op over the flat values of a ragged tensor.

    Args:
        *args (RaggedArg): Operation and input ragged tensor.
        **kwargs (RaggedKwarg): Additional keyword arguments.

    Returns:
        Tensor | tuple[Tensor, ...]: Ragged tensor with mapped values.
    """
    return _ragged_op("MapFlatValues", *args, **kwargs)
