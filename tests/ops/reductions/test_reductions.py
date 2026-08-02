# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.reductions.frontend as rd
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig


def test_reductions_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        assert rd.sum(t) is not None
        assert rd.max(t) is not None
        assert rd.min(t) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        import ml_switcheroo_compiler.ops.shape.utils as shape_utils

        orig_emit = shape_utils._emit_shape_node

        class DummyOp:
            def __call__(self, *args, **kwargs):
                return "reduction"

            def infer_shape(self, *args, **kwargs):
                return ()

        shape_utils._emit_shape_node = lambda *args, **kwargs: DummyOp()
        try:
            assert rd.sum(t) is not None
            assert rd.max(t) is not None
            assert rd.min(t) is not None
        finally:
            shape_utils._emit_shape_node = orig_emit
            config.eager_mode = True
