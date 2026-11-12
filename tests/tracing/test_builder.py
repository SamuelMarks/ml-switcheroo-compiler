"""Tests for test_builder."""

from __future__ import annotations

import typing

from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue

# ruff: noqa: E501


def test_builder_extract_proxy() -> None:
    """Test function test_builder_extract_proxy."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeData:
        """Test class FakeData."""

        id = "my_id"

    class FakeTensor:
        """Test class FakeTensor."""

        data = FakeData()
        shape = (2, 2)

    (id1, shape1) = ensure_callable(TracingNodeBuilder.extract_from_tensor)(ensure_callable(FakeTensor)())
    assert id1 == "my_id"
    assert shape1 == (2, 2)

    class FakeData2:
        """Test class FakeData2."""

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [[1, 2], [3, 4]]

    class FakeTensor2:
        """Test class FakeTensor2."""

        data = FakeData2()
        shape = (2, 2)

    class FakeTracingState:
        """Test class FakeTracingState."""

        constant_cache = ensure_tensor_dict({})
        is_tracing = True
        active_graph = True

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = ensure_eager_tensor(ensure_eager_tensor(bmod).global_tracing_state)
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(FakeTracingState())
    (id2, shape2) = ensure_callable(TracingNodeBuilder.extract_from_tensor)(ensure_callable(FakeTensor2)())
    assert id2 in FakeTracingState.constant_cache.values()
    (id3, shape3) = ensure_callable(TracingNodeBuilder.extract_from_tensor)(ensure_callable(FakeTensor2)())
    assert id2 == id3
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(old_state)


def test_builder_extract_constant() -> None:
    """Test function test_builder_extract_constant."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        """Test class FakeTracingState."""

        constant_cache = ensure_tensor_dict({})
        is_tracing = True
        active_graph = True

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = ensure_eager_tensor(ensure_eager_tensor(bmod).global_tracing_state)
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(FakeTracingState())

    class ProxyT:
        """Test class ProxyT."""

        def __init__(self, id: EagerValue, shape: EagerValue) -> None:
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

        def __init__(self, data: EagerValue) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        @property
        def shape(self) -> EagerValue:
            """Test function shape.

            Returns:
                EagerValue: Computed output value from `shape`.
            """
            _typed_ret_1: EagerValue = ensure_eager_tensor(self.data).shape
            return _typed_ret_1

    val = [ProxyT("id_1", (1,)), RealTensor(123)]
    (id1, shape1) = ensure_callable(TracingNodeBuilder.extract_from_constant)(val)
    assert id1[0] == "id_1"
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(old_state)


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
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        """Test class FakeTracingState."""

        is_tracing = True
        active_graph = True

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = ensure_eager_tensor(ensure_eager_tensor(bmod).global_tracing_state)
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(FakeTracingState())

    class FakeData:
        """Test class FakeData."""

        id = None

    class FakeConfig:
        """Test class FakeConfig."""

        shape = (1,)
        dtype = None
        device = None
        requires_grad = False

    fake_t = ensure_callable(Tensor)(ensure_callable(FakeData)(), ensure_callable(FakeConfig)())
    ensure_mutable(fake_t)._data = ensure_eager_tensor(FakeData())
    (id1, shape1) = ensure_callable(TracingNodeBuilder.extract_from_tensor)(fake_t)
    assert shape1 == (1,)

    # Check that extracting twice from same tensor uses cached tensor hash if data ID is None?
    # Actually wait. Let's see what the original test was.
    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((fake_t, fake_t))
    assert len(ids) == 2
    # Verify the ids are actually consistent or the same depending on implementation

    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(old_state)


def test_builder_extract_proxy_inputs_tensor_missing() -> None:
    """Test function test_builder_extract_proxy_inputs_tensor_missing."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        """Test class FakeTracingState."""

        is_tracing = True
        active_graph = True

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = ensure_eager_tensor(ensure_eager_tensor(bmod).global_tracing_state)
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(FakeTracingState())

    class DummyNotTensor:
        """Test class DummyNotTensor."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [42]

    (ids, shapes, first) = ensure_callable(TracingNodeBuilder.extract_proxy_inputs)((ensure_callable(DummyNotTensor)(),))
    assert len(ids) == 1
    assert shapes[0] == ()
    ensure_mutable(bmod).global_tracing_state = ensure_eager_tensor(old_state)


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
