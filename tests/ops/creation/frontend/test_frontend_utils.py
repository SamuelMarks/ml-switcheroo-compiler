"""Tests for test_frontend_utils."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_creation_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_frontend_utils() -> None:
    """Test function test_frontend_utils."""
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    pass
    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant() -> None:
    """Test function test_frontend_utils_constant."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_constant_node

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, value: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                value: Test parameter `value`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                ndim = 1

                def tolist(self) -> EagerValue:
                    """Test function tolist."""
                    return [value]

            return T()

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_constant_node(0, DType.Float32)
    pass
    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_constant_node(0, DType.Float32)
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant_scalar() -> None:
    """Test function test_frontend_utils_constant_scalar."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_constant_node

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, value: EagerValue, dtype: EagerValue = None) -> EagerValue:
            """Test function array.

            Args:
                value: Test parameter `value`.
                dtype: Test parameter `dtype`.
            """

            class T:
                """Test class T."""

                shape = ()
                ndim = 0

                def item(self) -> EagerValue:
                    """Test function item."""
                    return value

            return T()

    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_constant_node(0, DType.Float32)
    assert callable(_emit_constant_node)


def test_frontend_utils_missing_3() -> None:
    """Test function test_frontend_utils_missing_3."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        """Test class MockTensor."""

        shape = (10,)

    assert typing.cast("EagerCallableProtocol", _multinomial_shape)(typing.cast("EagerCallableProtocol", MockTensor)(), size=5) == (5,)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)


def test_frontend_utils_frompyfunc() -> None:
    """Test function test_frontend_utils_frompyfunc."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import frompyfunc

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "dummy_frompyfunc"
        assert frompyfunc(lambda x: x, 1, 1) == "dummy_frompyfunc"
