from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import (
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_frontend_pool_edge_cases():
    global_tracing_state.is_tracing = True
    try:
        from ml_switcheroo_ir import LogicalGraph

        graph = LogicalGraph()
        global_tracing_state.active_graph = graph

        class DummyData:
            id = "dummy1"

        t1d = Tensor(DummyData(), TensorConfig((1,), "float32", None))

        fractional_max_pool2d(t1d, output_size=[1, 1])
        adaptive_avg_pool2d(t1d, output_size=[1, 1])
        adaptive_max_pool2d(t1d, output_size=[1, 1])

        class DummyData:
            id = "dummy2"

        t2d = Tensor(DummyData(), TensorConfig((1, 1), "float32", None))

        fractional_max_pool3d(t2d, output_size=[1, 1, 1])
        adaptive_avg_pool3d(t2d, output_size=[1, 1, 1])
        adaptive_max_pool3d(t2d, output_size=[1, 1, 1])

    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
