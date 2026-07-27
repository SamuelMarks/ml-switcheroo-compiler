from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.nn.activations import one_hot


def test_one_hot_shape_metadata():
    orig = config.eager_mode
    config.eager_mode = False

    class MockMeta:
        def __init__(self):
            self.shape = (2,)

    class DummyTensor:
        shape_metadata = MockMeta()
        id = "t1"
        shape = (2,)

    try:
        import ml_switcheroo_compiler.tracing.state as state

        orig_tracing = state.global_tracing_state.is_tracing
        state.global_tracing_state.is_tracing = True
        orig_add_node = state.global_tracing_state.add_node
        state.global_tracing_state.add_node = lambda node: None
        res = one_hot(DummyTensor(), 5)
        assert res is not None
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
    finally:
        config.eager_mode = orig
