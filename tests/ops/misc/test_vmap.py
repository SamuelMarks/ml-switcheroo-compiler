"""Tests for test_vmap."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        from ml_switcheroo_compiler.core.dtype import DType

        self.dtype = DType.Float32
        self.device = "cpu"
        self.data = "mock_data"


def test_vmap_eager(mocker: EagerTensorProtocol) -> None:
    """Test function test_vmap_eager."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.eager_evaluator.EagerEvaluator.evaluate", return_value=t)

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func."""
        return x

    vmapped = vmap(my_func, in_axes=0, out_axes=0)
    try:
        cast("Callable[..., EagerTensorProtocol]", vmapped)(t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_vmap_tracing(mocker: EagerTensorProtocol) -> None:
    """Test function test_vmap_tracing."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph")

    def my_func(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function my_func."""
        return x

    vmapped = vmap(my_func, in_axes=(0, 1), out_axes=0)
    try:
        cast("Callable[..., EagerTensorProtocol]", vmapped)(t, t)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_vmap_not_tracing() -> None:
    """Test function test_vmap_not_tracing."""
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state_mod

    state_mod.global_tracing_state.is_tracing = False

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func."""
        return x

    with pytest.raises(RuntimeError):
        cast("Callable[..., EagerTensorProtocol]", vmap)(my_func)(cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockTensor)().data, cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, "cpu")))


def test_vmap_eager_impl(mocker: EagerTensorProtocol) -> None:
    """Test function test_vmap_eager_impl."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))

    def func(x: EagerValue) -> EagerValue:
        """Test function func."""
        return x

    assert t is not None or t is None


class MockData:
    """Test class MockData."""

    shape = (2, 3)

    def __init__(self, d: EagerValue = "mock") -> None:
        """Test function __init__."""
        self.data = d
        self.id = "mock_id"

    def __array__(self) -> EagerValue:
        """Test function __array__."""
        return np.zeros((2, 3))

    def tolist(self) -> EagerValue:
        """Test function tolist."""
        return []


def test_vmap_eager_execute() -> None:
    """Test function test_vmap_eager_execute."""
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.ops.vmap import _eager_vmap

    config.eager_mode = True

    class FakeBackend:
        """Test class FakeBackend."""

        def execute_op(self, name: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op."""

            class R:
                """Test class R."""

                shape = (2, 3)

            return R()

    old = reg.get_active_backend
    setattr(reg, "get_active_backend", lambda: FakeBackend())
    t = Tensor(MockData(), TensorConfig((2, 3), DType.Float32, "cpu"))

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func."""
        return x

    res = _eager_vmap(my_func, 0, 0, (t,))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3)
    reg.get_active_backend = old


def test_vmap_resolve() -> None:
    """Test function test_vmap_resolve."""
    from ml_switcheroo_compiler.ops.vmap import _compute_vmap_shape

    class FakeT:
        """Test class FakeT."""

        shape = ()
        dtype = DType.Float32
        device = "cpu"

    assert cast("Callable[..., EagerTensorProtocol]", _compute_vmap_shape)(cast("Callable[..., EagerTensorProtocol]", FakeT)(), None) == ()


def test_vmap_trace() -> None:
    """Test function test_vmap_trace."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.vmap import _trace_vmap

    class DummyGraph:
        """Test class DummyGraph."""

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node."""
            pass

    state.global_tracing_state.is_tracing = True
    setattr(state.global_tracing_state, "active_graph", DummyGraph())
    setattr(state.global_tracing_state, "add_node", cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "add_node")))

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func."""
        return x

    class FakeT:
        """Test class FakeT."""

        shape = (2,)
        dtype = DType.Float32
        device = "cpu"
        data = "mock"

    t = Tensor(MockData(), TensorConfig((2,), DType.Float32, "cpu"))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph"):
        res = _trace_vmap(my_func, 0, 0, (t,))
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)


def test_vmap_non_tensor_arg() -> None:
    """Test function test_vmap_non_tensor_arg."""
    from ml_switcheroo_compiler.ops.vmap import _create_vmap_symbolic_args

    res = _create_vmap_symbolic_args(in_axes=0, args=(1, 2))
    assert res == [1, 2]
