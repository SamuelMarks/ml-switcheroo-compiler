import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import Generator, stateless_split
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_random_stateless_missing():
    gen = Generator()

    with ConfigContext(eager_mode=False):
        original_tracing = global_tracing_state.is_tracing
        original_graph = global_tracing_state.active_graph
        try:
            global_tracing_state.is_tracing = True

            class DummyGraph:
                name = "dummy"
                nodes = {}

                def add_node(self, node):
                    pass

            global_tracing_state.active_graph = DummyGraph()

            # normal
            gen.normal((2,))

            # uniform
            gen.uniform((2,))

            # stateless_split
            seed_tensor = create_eager_tensor(np.array([42, 0]))
            stateless_split(seed_tensor, num=2)

        finally:
            global_tracing_state.is_tracing = original_tracing
            global_tracing_state.active_graph = original_graph
