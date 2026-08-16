# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Ragged frontends."""

from typing import Any

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


def _ragged_op(op_name: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ragged_op operation.

    Args:
        op_name (object): The op_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
        [getattr(args[0], "dtype", "float32") if args else "float32"],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    )


def ragged_constant(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_constant operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedConstant", *args, **kwargs)


def ragged_cross(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_cross operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias


def ragged_cross_hashed(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_cross_hashed operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_range(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_range operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedRange", *args, **kwargs)


def ragged_row_splits_to_segment_ids(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_row_splits_to_segment_ids operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)


def ragged_segment_ids_to_row_splits(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_segment_ids_to_row_splits operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)


def ragged_stack(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_stack operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedStack", *args, **kwargs)


def ragged_stack_dynamic_partitions(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_stack_dynamic_partitions operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)


def ragged_dot(*args: Any, **kwargs: Any) -> Any:
    """Evaluate ragged_dot operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("RaggedDot", *args, **kwargs)


def boolean_mask(*args: Any, **kwargs: Any) -> Any:
    """Boolean mask.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("BooleanMask", *args, **kwargs)


def map_flat_values(*args: Any, **kwargs: Any) -> Any:
    """Map flat values.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _ragged_op("MapFlatValues", *args, **kwargs)
