"""Tests for test_frontend_matrix."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.creation.frontend_matrix import _diag_eager, diag, eye, identity
from tests.testing_utils import as_callable, as_tensor, as_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_frontend_matrix() -> None:
    """Test function test_frontend_matrix."""
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

                shape = (1, 1)

            return T()

    import ml_switcheroo_compiler.backends.registry as reg

    orig_reg_backend = getattr(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = hasattr(reg, "_ACTIVE_BACKEND")
    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    import ml_switcheroo_compiler.ops.creation.frontend_matrix

    orig_frontend_matrix_backend = as_tensor(getattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend"))
    setattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend", lambda: DummyBackend())

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert as_tensor(getattr(eye(1), "shape")) == (1, 1)
    assert as_tensor(getattr(identity(1), "shape")) == (1, 1)

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

    assert as_tensor(getattr(as_callable(_diag_eager)(as_callable(DummyInput)(), 0, "cpu", "float32"), "shape")) == (1, 1)
    assert as_tensor(getattr(as_callable(_diag_eager)(as_callable(DummyInputNoDtype)(), 0, "cpu", None), "shape")) == (1, 1)

    assert as_tensor(getattr(as_callable(diag)(as_callable(DummyInput)()), "shape")) == (1, 1)

    class DummyInput2:
        """Test class DummyInput2."""

        data = type("D", (), {"id": "id"})()
        device = "cpu"
        dtype = "float32"
        shape = (2, 2)

    config.eager_mode = False
    with pytest.raises(RuntimeError):
        as_callable(diag)(as_callable(DummyInput2)())

    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = as_tensor_dict({})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", as_tensor(getattr(state.global_tracing_state.active_graph, "add_node")))

    assert as_tensor(getattr(as_callable(diag)(as_callable(DummyInput2)()), "shape")) == (2,)

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
        as_callable(diag)(as_callable(DummyInput3)())

    if has_reg_backend:
        setattr(reg, "_ACTIVE_BACKEND", orig_reg_backend)
    else:
        delattr(reg, "_ACTIVE_BACKEND")
    setattr(ml_switcheroo_compiler.ops.creation.frontend_matrix, "get_active_backend", orig_frontend_matrix_backend)
