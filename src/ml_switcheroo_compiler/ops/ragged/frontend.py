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

# pragma: no cover
from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _ragged_op(op_name: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    # pragma: no cover
    op_cls = get_op(op_name)  # pragma: no cover
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(op_name, *[getattr(a, "data", a) for a in args], **kwargs)  # pragma: no cover
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


def ragged_constant(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedConstant", *args, **kwargs)  # pragma: no cover


def ragged_cross(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # Alias  # pragma: no cover


def ragged_cross_hashed(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedCrossHashed", *args, **kwargs)  # pragma: no cover


def ragged_range(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedRange", *args, **kwargs)  # pragma: no cover


def ragged_row_splits_to_segment_ids(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedRowSplitsToSegmentIds", *args, **kwargs)  # pragma: no cover


def ragged_segment_ids_to_row_splits(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedSegmentIdsToRowSplits", *args, **kwargs)  # pragma: no cover


def ragged_stack(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedStack", *args, **kwargs)  # pragma: no cover


def ragged_stack_dynamic_partitions(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return _ragged_op("RaggedStackDynamicPartitions", *args, **kwargs)  # pragma: no cover


def ragged_dot(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    # pragma: no cover
    # pragma: no cover
    return _ragged_op("RaggedDot", *args, **kwargs)
