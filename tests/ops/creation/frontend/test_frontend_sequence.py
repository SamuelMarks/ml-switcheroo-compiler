"""Tests for test_frontend_sequence."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_sequence import arange, linspace
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_frontend_sequence() -> None:
    """Test function test_frontend_sequence."""
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

            Returns:
                EagerValue: Computed output value from `execute_op`.
            """

            class T:
                """Test class T."""

                shape = (1,)

            return T()

    import ml_switcheroo_compiler.ops.creation.frontend_sequence

    orig_backend = ensure_eager_tensor(ensure_eager_tensor(ml_switcheroo_compiler.ops.creation.frontend_sequence).get_active_backend)
    ensure_mutable(ml_switcheroo_compiler.ops.creation.frontend_sequence).get_active_backend = ensure_eager_tensor(lambda: DummyBackend())

    assert ensure_eager_tensor(arange(10)).shape == (10,)
    assert ensure_eager_tensor(arange(0, 10)).shape == (10,)
    assert ensure_eager_tensor(linspace(1, 10, 50)).shape == (50,)

    config.eager_mode = False

    with patch.object(ml_switcheroo_compiler.ops.creation.frontend_sequence, "_emit_creation_node", return_value="emitted"):
        assert arange(10) == "emitted"
        assert arange(0, 10) == "emitted"
        assert linspace(1, 10, 50) == "emitted"

    config.eager_mode = True
    ensure_mutable(ml_switcheroo_compiler.ops.creation.frontend_sequence).get_active_backend = ensure_eager_tensor(orig_backend)
