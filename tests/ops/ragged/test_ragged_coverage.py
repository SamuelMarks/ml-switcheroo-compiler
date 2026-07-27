# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.ragged.frontend as ragged_frontend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged import (
    BooleanMask,
    MapFlatValues,
    RaggedAdd,
    RaggedConstant,
    RaggedCrossHashed,
    RaggedDynamicBroadcast,
    RaggedGather,
    RaggedMatMul,
    RaggedRange,
    RaggedRowSplitsToSegmentIds,
    RaggedSegmentIdsToRowSplits,
    RaggedStack,
    RaggedStackDynamicPartitions,
    RaggedTensorToDense,
    boolean_mask,
    map_flat_values,
    ragged_tensor_to_dense,
)


def test_ragged_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    assert RaggedGather().infer_shape() == ()
    assert RaggedTensorToDense().infer_shape() == ()
    assert RaggedAdd().infer_shape() == ()
    assert RaggedMatMul().infer_shape() == ()
    assert RaggedDynamicBroadcast().infer_shape() == ()
    assert RaggedConstant().infer_shape() == ()
    assert RaggedCrossHashed().infer_shape() == ()
    assert RaggedRange().infer_shape() == ()
    assert RaggedRowSplitsToSegmentIds().infer_shape() == ()
    assert RaggedSegmentIdsToRowSplits().infer_shape() == ()
    assert RaggedStack().infer_shape() == ()
    assert RaggedStackDynamicPartitions().infer_shape() == ()
    assert BooleanMask().infer_shape() == ()
    assert MapFlatValues().infer_shape(t) == ()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert ragged_tensor_to_dense(t) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        assert ragged_tensor_to_dense(t) is not None
        config.eager_mode = True
        global_tracing_state.is_tracing = False

    class DummyOp:
        device = "cpu"

        def __init__(self):
            self.op_type = "dummy"

    orig_eager = config.eager_mode
    orig_get_op = ragged_frontend.get_op
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        class DummyOpMask:
            def __call__(self, *args, **kwargs):
                return "masked"

            def infer_shape(self, *args, **kwargs):
                return ()

        ragged_frontend.get_op = lambda x: DummyOpMask
        import ml_switcheroo_compiler.ops.linalg.utils as linalg_utils

        orig_emit = linalg_utils._emit_linalg_node
        linalg_utils._emit_linalg_node = lambda *args, **kwargs: "masked"
        try:
            assert boolean_mask(t, t) is not None
            assert map_flat_values(DummyOp(), t) is not None
        finally:
            linalg_utils._emit_linalg_node = orig_emit
    finally:
        config.eager_mode = orig_eager
        ragged_frontend.get_op = orig_get_op
        global_tracing_state.is_tracing = False
