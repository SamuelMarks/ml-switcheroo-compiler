# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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


def _ragged_op(op_name, *args, **kwargs):
    """Evaluate _ragged_op operation.

    Args:
        op_name (Any): The op_name parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
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


def ragged_constant(*args, **kwargs):
    """Evaluate ragged_constant operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedConstant", *args, **kwargs)


def ragged_cross(*args, **kwargs):
    """Evaluate ragged_cross operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias


def ragged_cross_hashed(*args, **kwargs):
    """Evaluate ragged_cross_hashed operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_range(*args, **kwargs):
    """Evaluate ragged_range operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedRange", *args, **kwargs)


def ragged_row_splits_to_segment_ids(*args, **kwargs):
    """Evaluate ragged_row_splits_to_segment_ids operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)


def ragged_segment_ids_to_row_splits(*args, **kwargs):
    """Evaluate ragged_segment_ids_to_row_splits operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)


def ragged_stack(*args, **kwargs):
    """Evaluate ragged_stack operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedStack", *args, **kwargs)


def ragged_stack_dynamic_partitions(*args, **kwargs):
    """Evaluate ragged_stack_dynamic_partitions operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)


def ragged_dot(*args, **kwargs):
    """Evaluate ragged_dot operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("RaggedDot", *args, **kwargs)


def boolean_mask(*args, **kwargs):
    """Boolean mask.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("BooleanMask", *args, **kwargs)


def map_flat_values(*args, **kwargs):
    """Map flat values.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _ragged_op("MapFlatValues", *args, **kwargs)
