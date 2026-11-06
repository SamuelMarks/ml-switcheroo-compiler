"""Tests for test_frontend_random."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_random import manual_seed, rand, randint, randn
from tests.testing_utils import ensure_eager_tensor, ensure_mutable, get_backend_member, has_backend_member

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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

    orig_reg_backend = get_backend_member(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = has_backend_member(reg, "_ACTIVE_BACKEND")
    ensure_mutable(reg)._ACTIVE_BACKEND = ensure_eager_tensor(DummyBackend())
    import ml_switcheroo_compiler.ops.creation.frontend_random as rand_mod

    orig_rand_backend = ensure_eager_tensor(ensure_eager_tensor(rand_mod).get_active_backend)
    ensure_mutable(rand_mod).get_active_backend = ensure_eager_tensor(lambda: DummyBackend())

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert ensure_eager_tensor(ensure_eager_tensor(rand(1)).data) == "Rand"
    assert ensure_eager_tensor(ensure_eager_tensor(randn(1)).data) == "Randn"
    assert ensure_eager_tensor(ensure_eager_tensor(randint(0, 10, (1,))).data) == "Randint"

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
        ensure_mutable(reg)._ACTIVE_BACKEND = ensure_eager_tensor(orig_reg_backend)
    else:
        del ensure_mutable(reg)._ACTIVE_BACKEND
    ensure_mutable(rand_mod).get_active_backend = ensure_eager_tensor(orig_rand_backend)
