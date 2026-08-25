import pytest


@pytest.fixture(autouse=True)
def reset_global_tracing_state():
    import ml_switcheroo_compiler.backends.registry as reg
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    old_is_tracing = state.global_tracing_state.is_tracing
    old_graph = state.global_tracing_state.active_graph
    old_add_node = state.global_tracing_state.add_node

    old_backend = config.backend
    old_eager = config.eager_mode
    had_active_backend = hasattr(reg, "_ACTIVE_BACKEND")
    old_active_backend = getattr(reg, "_ACTIVE_BACKEND", None)

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    orig_yaml_registry = _YAML_REGISTRY.copy()
    config.eager_mode = False
    config.backend = "numpy"

    yield

    _YAML_REGISTRY.clear()
    _YAML_REGISTRY.update(orig_yaml_registry)
    state.global_tracing_state.is_tracing = old_is_tracing
    state.global_tracing_state.active_graph = old_graph
    state.global_tracing_state.add_node = old_add_node

    config.backend = old_backend
    config.eager_mode = old_eager

    if had_active_backend:
        reg._ACTIVE_BACKEND = old_active_backend
    elif hasattr(reg, "_ACTIVE_BACKEND"):
        del reg._ACTIVE_BACKEND
