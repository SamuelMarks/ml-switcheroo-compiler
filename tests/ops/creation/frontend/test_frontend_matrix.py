"""Tests for test_frontend_matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.creation.frontend_matrix import _diag_eager, diag, eye, identity

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_frontend_matrix() -> None:
    """Test function test_frontend_matrix."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op."""

            class T:
                """Test class T."""

                shape = (1, 1)

            return T()

    import ml_switcheroo_compiler.backends.registry as reg

    orig_reg_backend = getattr(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = hasattr(reg, "_ACTIVE_BACKEND")
    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    import ml_switcheroo_compiler.ops.creation.frontend_matrix

    orig_frontend_matrix_backend = cast("EagerTensorProtocol", getattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend"))
    setattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend", lambda: DummyBackend())

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert cast("EagerTensorProtocol", getattr(eye(1), "shape")) == (1, 1)
    assert cast("EagerTensorProtocol", getattr(identity(1), "shape")) == (1, 1)

    class DummyInput:
        """Test class DummyInput."""

        data = "data"
        device = "cpu"
        dtype = "float32"
        shape = (1,)

    class DummyInputNoDtype:
        """Test class DummyInputNoDtype."""

        data = "data"
        device = "cpu"
        dtype = None
        shape = (1,)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", _diag_eager)(cast("Callable[..., EagerTensorProtocol]", DummyInput)(), 0, "cpu", "float32"), "shape")) == (1, 1)
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", _diag_eager)(cast("Callable[..., EagerTensorProtocol]", DummyInputNoDtype)(), 0, "cpu", None), "shape")) == (1, 1)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", diag)(cast("Callable[..., EagerTensorProtocol]", DummyInput)()), "shape")) == (1, 1)

    class DummyInput2:
        """Test class DummyInput2."""

        data = type("D", (), {"id": "id"})()
        device = "cpu"
        dtype = "float32"
        shape = (2, 2)

    config.eager_mode = False
    with pytest.raises(RuntimeError):
        cast("Callable[..., EagerTensorProtocol]", diag)(cast("Callable[..., EagerTensorProtocol]", DummyInput2)())

    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node."""
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", diag)(cast("Callable[..., EagerTensorProtocol]", DummyInput2)()), "shape")) == (2,)

    with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix._emit_creation_node", return_value="emitted"):
        assert eye(1) == "emitted"

    config.eager_mode = True

    class DummyInput3:
        """Test class DummyInput3."""

        data = type("D", (), {"id": "id"})()
        device = "cpu"
        dtype = "float32"
        shape = (1, 1, 1)

    config.eager_mode = False
    with pytest.raises(ValueError):
        cast("Callable[..., EagerTensorProtocol]", diag)(cast("Callable[..., EagerTensorProtocol]", DummyInput3)())

    if has_reg_backend:
        setattr(reg, "_ACTIVE_BACKEND", orig_reg_backend)
    else:
        delattr(reg, "_ACTIVE_BACKEND")
    setattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend", orig_frontend_matrix_backend)
