# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103
"""Ragged frontends."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# from ml_switcheroo_compiler.ops.ragged import (
#    RaggedConstant,
#    RaggedCrossHashed,
#    RaggedRange,
#    RaggedRowSplitsToSegmentIds,
#    RaggedSegmentIdsToRowSplits,
#    RaggedStack,
##    RaggedStackDynamicPartitions,
# )
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node


def _ragged_op(op_name, *args, **kwargs):
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op

    op_cls = get_op(op_name)
    if config.eager_mode:  # pragma: no cover
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
    return _ragged_op("RaggedConstant", *args, **kwargs)


def ragged_cross(*args, **kwargs):
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias


def ragged_cross_hashed(*args, **kwargs):
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)


def ragged_range(*args, **kwargs):
    return _ragged_op("RaggedRange", *args, **kwargs)


def ragged_row_splits_to_segment_ids(*args, **kwargs):
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)


def ragged_segment_ids_to_row_splits(*args, **kwargs):
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)


def ragged_stack(*args, **kwargs):
    return _ragged_op("RaggedStack", *args, **kwargs)


def ragged_stack_dynamic_partitions(*args, **kwargs):
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)
