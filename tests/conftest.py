"""Global pytest fixtures and test environment initialization."""

from __future__ import annotations

import importlib.util
import sys
import typing

import numpy as np
import pytest
import yaml

yaml.safe_load("dummy: 1")
sys.modules.setdefault("dpnp", np)

for _opt_pkg in ("scipy.linalg", "jax", "dask", "jinja2", "torch", "numba", "sparse"):
    if importlib.util.find_spec(_opt_pkg) is not None:
        __import__(_opt_pkg)

if importlib.util.find_spec("tensorflow") is not None:
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_global_tracing_state() -> Iterator[None]:
    """Test function reset_global_tracing_state."""
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

    import copy

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY
    from ml_switcheroo_compiler.utils.generic_utils import get_custom_objects

    for _tmp_file in ("/tmp/ml_switcheroo_test_file.txt", "/tmp/ml_switcheroo_test_file2.txt"):
        with open(_tmp_file, "wb") as _f:
            _f.write(b"content")

    get_custom_objects().clear()
    orig_yaml_registry = copy.deepcopy(_YAML_REGISTRY)
    config.eager_mode = False
    config.backend = "numpy"

    yield

    get_custom_objects().clear()
    _YAML_REGISTRY.clear()
    _YAML_REGISTRY.update(orig_yaml_registry)
    state.global_tracing_state.is_tracing = old_is_tracing
    state.global_tracing_state.active_graph = old_graph
    setattr(state.global_tracing_state, "add_node", old_add_node)

    config.backend = old_backend
    config.eager_mode = old_eager

    if had_active_backend:
        setattr(reg, "_ACTIVE_BACKEND", old_active_backend)
    elif hasattr(reg, "_ACTIVE_BACKEND"):
        del reg._ACTIVE_BACKEND
