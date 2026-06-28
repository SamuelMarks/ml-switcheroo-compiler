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
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    # pragma: no cover
    op_cls = get_op(op_name)  # pragma: no cover
    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            op_name, *[getattr(a, "data", a) for a in args], **kwargs
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            data,  # pragma: no cover
            TensorConfig(  # pragma: no cover
                getattr(data, "shape", ()),  # pragma: no cover
                getattr(args[0], "dtype", "float32") if args else "float32",  # pragma: no cover
                getattr(args[0], "device", "cpu") if args else "cpu",  # pragma: no cover
            ),  # pragma: no cover
        )  # pragma: no cover
    # pragma: no cover
    op = op_cls()  # pragma: no cover
    out_shape = op.infer_shape(*args, **kwargs)  # pragma: no cover
    return _emit_linalg_node(  # pragma: no cover
        op_name,
        list(args),
        kwargs,
        [tuple(out_shape)],
        [getattr(args[0], "dtype", "float32") if args else "float32"],
    )


def ragged_constant(*args, **kwargs):
    return _ragged_op("RaggedConstant", *args, **kwargs)  # pragma: no cover


def ragged_cross(*args, **kwargs):
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias  # pragma: no cover


def ragged_cross_hashed(*args, **kwargs):
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # pragma: no cover


def ragged_range(*args, **kwargs):
    return _ragged_op("RaggedRange", *args, **kwargs)  # pragma: no cover


def ragged_row_splits_to_segment_ids(*args, **kwargs):
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)  # pragma: no cover


def ragged_segment_ids_to_row_splits(*args, **kwargs):
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)  # pragma: no cover


def ragged_stack(*args, **kwargs):
    return _ragged_op("RaggedStack", *args, **kwargs)  # pragma: no cover


def ragged_stack_dynamic_partitions(*args, **kwargs):
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)  # pragma: no cover


def ragged_dot(*args, **kwargs):
    # pragma: no cover
    # pragma: no cover
    return _ragged_op("RaggedDot", *args, **kwargs)
