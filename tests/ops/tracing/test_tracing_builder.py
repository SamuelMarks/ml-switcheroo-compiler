"""Tests for test_tracing_builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_extract_from_constant(mocker: EagerTensorProtocol) -> None:
    """Test function test_extract_from_constant.

    Args:
        mocker: Test parameter `mocker`.
    """

    class Proxy:
        """Test class Proxy."""

        def __init__(self, id: EagerValue, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
                shape: Test parameter `shape`.
            """
            self.id = id
            self.shape = shape

    Proxy.__name__ = "ProxyTensor"
    p = type("ProxyTensor", (), {"id": "1", "shape": (2,)})()
    assert TracingNodeBuilder.extract_from_constant([p]) == (["1"], [(2,)])
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    assert TracingNodeBuilder.extract_from_constant([1, p])[1] == [(), (2,)]


def test_extract_from_tensor(mocker: EagerTensorProtocol) -> None:
    """Test function test_extract_from_tensor.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")

    class MockTensor1:
        """Test class MockTensor1."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = type("M", (), {"id": "1"})()
            self.shape = (2,)

    assert cast("EagerCallableProtocol", TracingNodeBuilder.extract_from_tensor)(cast("EagerCallableProtocol", MockTensor1)()) == ("1", (2,))

    class MockTensor2:
        """Test class MockTensor2."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = 42
            self.shape = ()

    res = cast("EagerCallableProtocol", TracingNodeBuilder.extract_from_tensor)(cast("EagerCallableProtocol", MockTensor2)())
    assert res[1] == ()

    # Test the branch where hasattr(global_tracing_state, "constant_cache") is False
    from unittest.mock import patch

    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    original_hasattr = hasattr

    def custom_hasattr(obj: EagerValue, name: EagerValue) -> EagerValue:
        """Test function custom_hasattr.

        Args:
            obj: Test parameter `obj`.
            name: Test parameter `name`.
        """
        if cast("EagerTensorProtocol", obj) is cast("EagerTensorProtocol", global_tracing_state) and cast("EagerTensorProtocol", name) == cast("EagerTensorProtocol", "constant_cache"):
            return False
        return cast("EagerCallableProtocol", original_hasattr)(obj, name)

    with patch("builtins.hasattr", side_effect=custom_hasattr):
        res2 = cast("EagerCallableProtocol", TracingNodeBuilder.extract_from_tensor)(cast("EagerCallableProtocol", MockTensor2)())
        assert res2[1] == ()

    # Hit the cache condition
    m2 = MockTensor2()
    with patch.object(global_tracing_state, "constant_cache", {id(m2.data): "cached_id"}, create=True):
        res3 = cast("EagerCallableProtocol", TracingNodeBuilder.extract_from_tensor)(m2)
        assert res3 == ("cached_id", ())


def test_extract_proxy_inputs(mocker: EagerTensorProtocol) -> None:
    """Test function test_extract_proxy_inputs.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")

    class M:
        """Test class M."""

        id = "id"
        shape = ()

    t = Tensor(M(), TensorConfig((), "float32", "cpu"))
    (ids, shapes, ft) = TracingNodeBuilder.extract_proxy_inputs((t, M(), 1))
    assert ids[0] == "id"
    assert ids[1] == "id"
    assert ft is t


def test_create_tracing_logical_node(mocker: EagerTensorProtocol) -> None:
    """Test function test_create_tracing_logical_node.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mock_add = mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    out_id = TracingNodeBuilder.create_tracing_logical_node("Op", ["1"], {"a": 1}, (2, 3))
    assert type(out_id) is str
    mock_add.assert_called_once()


def test_emit_tracing_node(mocker: EagerTensorProtocol) -> None:
    """Test function test_emit_tracing_node.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.tracing.builder.infer_shape", return_value=(2, 3))

    class M:
        """Test class M."""

        id = "1"
        shape = ()
        dtype = "float32"
        device = "cpu"

    t = Tensor(M(), TensorConfig((), "float32", "cpu"))
    res = TracingNodeBuilder.emit_tracing_node("Op", t)
    assert res.config.shape == (2, 3)
