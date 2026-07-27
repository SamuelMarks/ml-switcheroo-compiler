"""Ragged frontends."""

# from ml_switcheroo_compiler.ops.ragged import (
#    RaggedConstant,
#    RaggedCrossHashed,
#    RaggedRange,
#    RaggedRowSplitsToSegmentIds,
#    RaggedSegmentIdsToRowSplits,
#    RaggedStack,
##    RaggedStackDynamicPartitions,
# )
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _ragged_op(op_name: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged op operation.

    Args:
        op_name (object): Required parameter for op_name.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
    return _emit_linalg_node(
        op_name,
        list(args),
        kwargs,
        [tuple(out_shape)],
        [getattr(args[0], "dtype", "float32") if args else "float32"],
    )


def ragged_constant(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged constant operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedConstant", *args, **kwargs)


def ragged_cross(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged cross operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias


def ragged_cross_hashed(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged cross hashed operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_range(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged range operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedRange", *args, **kwargs)


def ragged_row_splits_to_segment_ids(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged row splits to segment ids operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)


def ragged_segment_ids_to_row_splits(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged segment ids to row splits operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)


def ragged_stack(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged stack operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedStack", *args, **kwargs)


def ragged_stack_dynamic_partitions(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged stack dynamic partitions operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)


def ragged_dot(*args: object, **kwargs: object) -> object:
    """Evaluate and process the ragged dot operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _ragged_op("RaggedDot", *args, **kwargs)


def boolean_mask(*args: object, **kwargs: object) -> object:
    """Boolean mask."""
    return _ragged_op("BooleanMask", *args, **kwargs)


def map_flat_values(*args: object, **kwargs: object) -> object:
    """Map flat values."""
    return _ragged_op("MapFlatValues", *args, **kwargs)
