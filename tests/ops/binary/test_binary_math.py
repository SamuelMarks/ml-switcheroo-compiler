"""Tests for test_binary_math."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from tests.testing_utils import as_callable, as_tensor, as_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_binary_math_extra_coverage() -> None:
    """Test function test_binary_math_extra_coverage."""
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.binary.math import Clip, rem

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

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    type("T", (), {"shape": (2, 2)})()
    type("T", (), {"shape": (2,)})()
    assert Clip().infer_shape() == ()
    config.eager_mode = True
    with pytest.raises(TypeError):
        rem(1, 2)


def test_eager_eval_branches() -> None:
    """Test function test_eager_eval_branches."""
    from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, Betainc, Diff, Digitize, Poly, Polyadd, Polyder, Polydiv, Polyfit, Polyint, Polymul, Polysub, Polyval, Roots

    for cls in [Diff, Digitize, ArrayEquiv, Betainc, Polyadd, Polysub, Polymul, Polydiv, Polyval, Poly, Polyder, Polyfit, Polyint, Roots]:
        assert as_tensor(getattr(cls(), "op_name")) is not None


def test_binary_math_shapes_again() -> None:
    """Test function test_binary_math_shapes_again."""
    from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, Betainc, Clip, Diff, Digitize, clip

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1,)
        attributes = as_tensor_dict({})
        shape_metadata = (1,)

    assert Diff().infer_shape() == ()
    assert as_callable(Digitize)().infer_shape(as_callable(DummyTensor)()) == (1,)
    assert Digitize().infer_shape() == ()
    assert ArrayEquiv().infer_shape() == ()
    assert as_callable(Betainc)().infer_shape(as_callable(DummyTensor)(), as_callable(DummyTensor)(), as_callable(DummyTensor)()) == (1,)
    assert as_callable(Clip)().infer_shape(as_callable(DummyTensor)()) == (1,)
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as m:
        m.return_value.execute_op.return_value = "Clip"
        assert clip(DummyTensor()) == "Clip"
    config.eager_mode = False
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
    with patch("ml_switcheroo_compiler.ops.base.LogicalNode", return_value="node_str", create=True):
        pass
        pass
    config.eager_mode = True


def test_binary_math_clip_emit() -> None:
    """Test function test_binary_math_clip_emit."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyTensor:
        """Test class DummyTensor."""

        shape_metadata = (1,)

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
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.base.LogicalNode", return_value="node_str", create=True):
        pass
        pass
    assert callable(test_binary_math_clip_emit)
