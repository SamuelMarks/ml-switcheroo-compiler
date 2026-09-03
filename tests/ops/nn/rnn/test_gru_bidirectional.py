from ml_switcheroo_compiler.ops.nn.gru import _compute_gru_gates, _sigmoid


def test_gru_extras():
    class DummyTensor:
        id = "t1"
        shape_metadata = None

        def __add__(self, other):
            return self

        def __mul__(self, other):
            return self

        def __sub__(self, other):
            return self

        def __rsub__(self, other):
            return self

    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        _compute_gru_gates([DummyTensor(), DummyTensor(), DummyTensor()], [DummyTensor(), DummyTensor(), DummyTensor()], DummyTensor())
        _sigmoid(DummyTensor())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
