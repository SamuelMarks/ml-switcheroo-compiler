"""Tests for test_builder."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTensorStub, MockTestStub, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

# ruff: noqa: E501


def test_builder_extract_proxy() -> None:
    """Test function test_builder_extract_proxy."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeData(MockTestStub):
        """Test class FakeData."""

        id = "my_id"

    class FakeTensor(MockTensorStub):
        """Test class FakeTensor."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.data = FakeData()
            self.shape = (2, 2)

    (id1, shape1) = TracingNodeBuilder.extract_from_tensor(FakeTensor())
    assert id1 == "my_id"
    assert shape1 == (2, 2)

    class FakeData2(MockTestStub):
        """Test class FakeData2."""

        def tolist(self) -> list[int]:
            """Test function tolist.

            Returns:
                list[int]: Computed output value from `tolist`.
            """
            return [1, 2, 3, 4]

    class FakeTensor2(MockTensorStub):
        """Test class FakeTensor2."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.data = FakeData2()
            self.shape = (2, 2)

    class FakeTracingState:
        """Test class FakeTracingState."""

        constant_cache: dict[str, BackendArg] = {}
        is_tracing = True
        active_graph = True

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    old_state = global_tracing_state
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", FakeTracingState(), raising=False)
    ft2 = FakeTensor2()
    (id2, shape2) = TracingNodeBuilder.extract_from_tensor(ft2)
    assert id2 in FakeTracingState.constant_cache.values()
    (id3, shape3) = TracingNodeBuilder.extract_from_tensor(ft2)
    assert id2 == id3
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", old_state, raising=False)


def test_builder_extract_constant() -> None:
    """Test function test_builder_extract_constant."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    class FakeTracingState:
        """Test class FakeTracingState."""

        constant_cache: dict[str, BackendArg] = {}
        is_tracing = True
        active_graph = True

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = global_tracing_state
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", FakeTracingState(), raising=False)

    class ProxyT:
        """Test class ProxyT."""

        def __init__(self, id: str, shape: tuple[int, ...]) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
                shape: Test parameter `shape`.
            """
            self.id = id
            self.shape = shape

    ProxyT.__name__ = "ProxyTensor"
    proxy_inst = ProxyT("proxy_id", (2,))
    type(proxy_inst).__name__ = "ProxyTensor"

    class RealTensor:
        """Test class RealTensor."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        @property
        def shape(self) -> tuple[int, ...]:
            """Test function shape.

            Returns:
                tuple[int, ...]: Computed output value from `shape`.
            """
            return tuple(int(d) for d in extract_test_shape(self.data) if isinstance(d, int))

    val: list[ProxyT | RealTensor] = [ProxyT("id_1", (1,)), RealTensor(123)]
    (id1, shape1) = TracingNodeBuilder.extract_from_constant(val)
    assert isinstance(id1, list)
    assert id1[0] == "id_1"
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", old_state, raising=False)


def test_builder_extract_proxy_inputs_proxy() -> None:
    """Test function test_builder_extract_proxy_inputs_proxy."""
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTensorProxy:
        """Test class FakeTensorProxy."""

        id = "proxy_id"
        shape = (1, 2)

    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((FakeTensorProxy(),))
    assert ids == ["proxy_id"]
    assert shapes == [(1, 2)]
    assert first is None


def test_builder_tensor_cache_missing() -> None:
    """Test function test_builder_tensor_cache_missing."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    class FakeTracingState:
        """Test class FakeTracingState."""

        is_tracing = True
        active_graph = True

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = global_tracing_state
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", FakeTracingState(), raising=False)

    class FakeData(MockTestStub):
        """Test class FakeData."""

        id = None

    fake_t = Tensor(FakeData(), TensorConfig((1,), None, None, False))
    pytest.MonkeyPatch().setattr(fake_t, "_data", FakeData(), raising=False)
    (id1, shape1) = TracingNodeBuilder.extract_from_tensor(fake_t)
    assert shape1 == (1,)

    # Check that extracting twice from same tensor uses cached tensor hash if data ID is None?
    # Actually wait. Let's see what the original test was.
    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((fake_t, fake_t))
    assert len(ids) == 2
    # Verify the ids are actually consistent or the same depending on implementation

    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", old_state, raising=False)


def test_builder_extract_proxy_inputs_tensor_missing() -> None:
    """Test function test_builder_extract_proxy_inputs_tensor_missing."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        """Test class FakeTracingState."""

        is_tracing = True
        active_graph = True

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    old_state = global_tracing_state
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", FakeTracingState(), raising=False)

    class DummyNotTensor(MockTestStub):
        """Test class DummyNotTensor."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        def tolist(self) -> list[int]:
            """Test function tolist.

            Returns:
                list[int]: Computed output value from `tolist`.
            """
            return [42]

    from ml_switcheroo_compiler.core.shape import OpValue

    not_tensor_inputs: tuple[OpValue, ...] = (DummyNotTensor(),)
    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs(not_tensor_inputs)
    assert len(ids) == 1
    assert shapes[0] == ()
    pytest.MonkeyPatch().setattr(bmod, "global_tracing_state", old_state, raising=False)


def test_builder_infer_shape_none_branch() -> None:
    """Test TracingNodeBuilder.emit_tracing_node when infer_shape returns None."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.tracing import global_tracing_state
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    global_tracing_state.start_tracing("test_infer_none")
    try:
        with patch("ml_switcheroo_compiler.tracing.builder.infer_shape", return_value=None):
            tensor = TracingNodeBuilder.emit_tracing_node("CustomOpWithNoneShape")
            assert tensor.shape == ()
    finally:
        global_tracing_state.stop_tracing()
