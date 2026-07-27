# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged.frontend import ragged_constant, ragged_cross, ragged_cross_hashed, ragged_dot, ragged_range, ragged_row_splits_to_segment_ids, ragged_segment_ids_to_row_splits, ragged_stack, ragged_stack_dynamic_partitions


def test_ragged_frontend_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert ragged_constant(t) is not None
        assert ragged_cross(t) is not None
        assert ragged_cross_hashed(t) is not None
        assert ragged_range(t) is not None
        assert ragged_row_splits_to_segment_ids(t) is not None
        assert ragged_segment_ids_to_row_splits(t) is not None
        assert ragged_stack(t) is not None
        assert ragged_stack_dynamic_partitions(t) is not None
        assert ragged_dot(t) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        import ml_switcheroo_compiler.ops.linalg.utils as l_utils

        orig_emit = l_utils._emit_linalg_node
        l_utils._emit_linalg_node = lambda *args, **kwargs: "ragged"
        try:
            assert ragged_constant(t) is not None
            assert ragged_cross(t) is not None
        finally:
            l_utils._emit_linalg_node = orig_emit
            config.eager_mode = True
            global_tracing_state.is_tracing = False
