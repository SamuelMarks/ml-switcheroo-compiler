"""Tests for test_frontend_sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_sequence import arange, linspace

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
            """

            class T:
                """Test class T."""

                shape = (1,)

            return T()

    import ml_switcheroo_compiler.ops.creation.frontend_sequence

    orig_backend = cast("EagerTensorProtocol", getattr(ml_switcheroo_compiler.ops.creation.frontend_sequence, "get_active_backend"))
    setattr(ml_switcheroo_compiler.ops.creation.frontend_sequence, "get_active_backend", lambda: DummyBackend())

    assert cast("EagerTensorProtocol", getattr(arange(10), "shape")) == (10,)
    assert cast("EagerTensorProtocol", getattr(arange(0, 10), "shape")) == (10,)
    assert cast("EagerTensorProtocol", getattr(linspace(1, 10, 50), "shape")) == (50,)

    config.eager_mode = False

    with patch.object(ml_switcheroo_compiler.ops.creation.frontend_sequence, "_emit_creation_node", return_value="emitted"):
        assert arange(10) == "emitted"
        assert arange(0, 10) == "emitted"
        assert linspace(1, 10, 50) == "emitted"

    config.eager_mode = True
    setattr(ml_switcheroo_compiler.ops.creation.frontend_sequence, "get_active_backend", orig_backend)
