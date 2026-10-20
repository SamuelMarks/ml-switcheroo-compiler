"""Tests for test_frontend_random."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_random import manual_seed, rand, randint, randn

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_frontend_random() -> None:
    """Test function test_frontend_random."""
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return op

    orig_reg_backend = getattr(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = hasattr(reg, "_ACTIVE_BACKEND")
    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    import ml_switcheroo_compiler.ops.creation.frontend_random as rand_mod

    orig_rand_backend = typing.cast("EagerTensorProtocol", getattr(rand_mod, "get_active_backend"))
    setattr(rand_mod, "get_active_backend", lambda: DummyBackend())

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert typing.cast("EagerTensorProtocol", getattr(rand(1), "data")) == "Rand"
    assert typing.cast("EagerTensorProtocol", getattr(randn(1), "data")) == "Randn"
    assert typing.cast("EagerTensorProtocol", getattr(randint(0, 10, (1,)), "data")) == "Randint"

    manual_seed(42)

    config.eager_mode = False

    with patch.object(rand_mod, "_emit_creation_node", return_value="emitted"):
        assert rand(1) == "emitted"
        assert randn(1) == "emitted"
        assert randint(0, 10, (1,)) == "emitted"

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph") as graph:
            graph.nodes = {}
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node"):
                manual_seed(42)

    config.eager_mode = True
    if has_reg_backend:
        setattr(reg, "_ACTIVE_BACKEND", orig_reg_backend)
    else:
        delattr(reg, "_ACTIVE_BACKEND")
    setattr(rand_mod, "get_active_backend", orig_rand_backend)
