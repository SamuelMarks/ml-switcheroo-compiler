import pytest


@pytest.fixture(autouse=True)
def reset_global_tracing_state():
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    old_is_tracing = state.global_tracing_state.is_tracing
    old_graph = state.global_tracing_state.active_graph
    old_add_node = state.global_tracing_state.add_node

    old_backend = config.backend
    old_eager = config.eager_mode

    config.eager_mode = False
    config.backend = "numpy"

    yield

    state.global_tracing_state.is_tracing = old_is_tracing
    state.global_tracing_state.active_graph = old_graph
    state.global_tracing_state.add_node = old_add_node

    config.backend = old_backend
    config.eager_mode = old_eager
