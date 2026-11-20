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
    if importlib.util.find_spec(_opt_pkg.split(".")[0]) is not None:
        __import__(_opt_pkg)


if importlib.util.find_spec("tensorflow") is not None:
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_global_tracing_state() -> Iterator[None]:
    """Reset global tracing, backend, and distributed state around each test.

    Yields:
        None: Test execution context.
    """
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as np_dist
    import ml_switcheroo_compiler.backends.registry as reg
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
    from ml_switcheroo_compiler.core.config import config

    old_is_tracing, old_graph, old_add_node = (
        state.global_tracing_state.is_tracing,
        state.global_tracing_state.active_graph,
        state.global_tracing_state.add_node,
    )
    old_backend, old_eager = config.backend, config.eager_mode
    had_active_backend = adapt_backend_module(reg).has_symbol("_ACTIVE_BACKEND")
    old_active_backend = adapt_backend_module(reg).get_symbol("_ACTIVE_BACKEND", None)
    np_dist._tcp_dist_ctx = np_dist.TCPDistributedContext()

    import os

    import ml_switcheroo_compiler.ops as _preload_ops
    import ml_switcheroo_compiler.ops.base as _preload_ops_base
    import ml_switcheroo_compiler.ops.nn as _preload_ops_nn
    import ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider as ad_prov
    import ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry as jvp_reg
    import ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry as vjp_reg
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, _YAML_REGISTRY

    assert (_preload_ops, _preload_ops_base, _preload_ops_nn) is not None

    for _tmp_file in ("/tmp/ml_switcheroo_test_file.txt", "/tmp/ml_switcheroo_test_file2.txt"):
        if not os.path.exists(_tmp_file):
            with open(_tmp_file, "wb") as _f:
                _f.write(b"content")

    orig_yaml_registry = dict(_YAML_REGISTRY)
    orig_backends = dict(reg.BackendRegistry._registry)
    orig_vjp_reg, orig_jvp_reg = dict(vjp_reg._VJP_REGISTRY), dict(jvp_reg._JVP_REGISTRY)
    orig_caches = (
        vjp_reg.load_primitive_vjp_rules(),
        jvp_reg.load_primitive_jvp_rules(),
        ad_prov._get_autodiff_manifest(),
        ad_prov._get_modular_rules(),
    )
    config.eager_mode = False
    config.backend = "numpy"

    yield

    reg.BackendRegistry._registry.clear()
    reg.BackendRegistry._registry.update(orig_backends)
    _YAML_REGISTRY.clear()
    _YAML_REGISTRY.update(orig_yaml_registry)
    for _cg_key in [k for k in _OP_REGISTRY if k.startswith("CustomGradient_")]:
        _OP_REGISTRY.pop(_cg_key, None)
    vjp_reg._VJP_REGISTRY.clear()
    vjp_reg._VJP_REGISTRY.update(orig_vjp_reg)
    jvp_reg._JVP_REGISTRY.clear()
    jvp_reg._JVP_REGISTRY.update(orig_jvp_reg)
    (
        vjp_reg._DECLARATIVE_VJP_CACHE,
        jvp_reg._DECLARATIVE_JVP_CACHE,
        ad_prov._MANIFEST_MODEL_CACHE,
        ad_prov._MODULAR_RULES_CACHE,
    ) = orig_caches
    state.global_tracing_state.is_tracing = old_is_tracing
    state.global_tracing_state.active_graph = old_graph
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", old_add_node, raising=False)

    config.backend = old_backend
    config.eager_mode = old_eager

    if had_active_backend:
        pytest.MonkeyPatch().setattr(reg, "_ACTIVE_BACKEND", old_active_backend, raising=False)
    elif adapt_backend_module(reg).has_symbol("_ACTIVE_BACKEND"):
        pytest.MonkeyPatch().delattr(reg, "_ACTIVE_BACKEND")
    np_dist._tcp_dist_ctx = np_dist.TCPDistributedContext()
