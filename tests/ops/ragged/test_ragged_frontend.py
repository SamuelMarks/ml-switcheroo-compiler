"""Behavioral tests for ragged tensor API wrappers in eager and IR tracing modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged.api import (
    _ragged_op,
    boolean_mask,
    map_flat_values,
    ragged_add,
    ragged_constant,
    ragged_cross,
    ragged_cross_hashed,
    ragged_dot,
    ragged_dynamic_broadcast,
    ragged_gather,
    ragged_matmul,
    ragged_range,
    ragged_row_splits_to_segment_ids,
    ragged_segment_ids_to_row_splits,
    ragged_stack,
    ragged_stack_dynamic_partitions,
    ragged_tensor_to_dense,
)
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import ensure_tensor, require_callable


def _make_tensor(shape: tuple[int, ...] = (2, 3)) -> Tensor:
    """Construct a float32 CPU `Tensor` filled with ones.

    Args:
        shape (tuple[int, ...]): Target tensor shape.

    Returns:
        Tensor: Constructed `Tensor` instance.
    """
    return Tensor(np.ones(shape, dtype=np.float32), TensorConfig(shape, DType.Float32, Device("cpu")))


def test_ragged_api_eager_dispatch() -> None:
    """Verify eager dispatch for all ragged API operations including zero-argument invocations."""
    t = _make_tensor()
    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 3), dtype=np.float32)

    with (
        ConfigContext(eager_mode=True),
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend),
    ):
        assert isinstance(ensure_tensor(require_callable(ragged_constant)([[1, 2], [3]])), Tensor)
        assert isinstance(ragged_constant(t, 42, foo="bar"), Tensor)
        assert isinstance(ragged_cross(t, t), Tensor)
        assert isinstance(ragged_cross_hashed(t, t), Tensor)
        assert isinstance(ragged_range(0, 10, 2), Tensor)
        assert isinstance(ragged_row_splits_to_segment_ids(t), Tensor)
        assert isinstance(ragged_segment_ids_to_row_splits(t), Tensor)
        assert isinstance(ensure_tensor(require_callable(ragged_stack)([t, t])), Tensor)
        assert isinstance(ragged_stack_dynamic_partitions(t, t), Tensor)
        assert isinstance(ragged_gather(t, t), Tensor)
        assert isinstance(ragged_tensor_to_dense(t), Tensor)
        assert isinstance(ragged_add(t, t), Tensor)
        assert isinstance(ragged_matmul(t, t), Tensor)
        assert isinstance(ragged_dynamic_broadcast(t, (2, 3)), Tensor)
        assert isinstance(ragged_dot(t, t), Tensor)
        assert isinstance(boolean_mask(t, t), Tensor)
        assert isinstance(ensure_tensor(require_callable(map_flat_values)(lambda x: x, t)), Tensor)

        res_no_args = _ragged_op("RaggedConstant")
        assert isinstance(res_no_args, Tensor)
        assert res_no_args.dtype == DType.Float32


def test_ragged_api_tracing_ir_emission() -> None:
    """Verify all ragged API operations emit canonical IR `LogicalNode`s during tracing."""
    t = Tensor(ProxyTensor("rag_in", (2, 3), "float32"), TensorConfig((2, 3), DType.Float32, Device("cpu")))

    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("ragged_api_trace")
        try:
            outputs = [
                ensure_tensor(require_callable(ragged_constant)([[1, 2], [3]])),
                ragged_cross(t, t),
                ragged_cross_hashed(t, t),
                ragged_range(t),
                ragged_row_splits_to_segment_ids(t),
                ragged_segment_ids_to_row_splits(t),
                ragged_stack(t, t),
                ragged_stack_dynamic_partitions(t, t),
                ragged_gather(t, t),
                ragged_tensor_to_dense(t),
                ragged_add(t, t),
                ragged_matmul(t, t),
                ragged_dynamic_broadcast(t),
                ragged_dot(t, t),
                boolean_mask(t, t),
                map_flat_values(None, t),
                _ragged_op("RaggedConstant"),
            ]
        finally:
            tape.stop_tracing()

    assert all(isinstance(out, Tensor) for out in outputs)
    emitted_ops = {node.op_type for node in traced_graph.nodes.values()}
    assert {
        "RaggedConstant",
        "RaggedCrossHashed",
        "RaggedRange",
        "RaggedRowSplitsToSegmentIds",
        "RaggedSegmentIdsToRowSplits",
        "RaggedStack",
        "RaggedStackDynamicPartitions",
        "RaggedGather",
        "RaggedTensorToDense",
        "RaggedAdd",
        "RaggedMatMul",
        "RaggedDynamicBroadcast",
        "RaggedDot",
        "BooleanMask",
        "MapFlatValues",
    } <= emitted_ops
    assert len(traced_graph.nodes) == 17
