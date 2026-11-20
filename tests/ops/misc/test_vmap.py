"""Tests for test_vmap."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap
from tests.testing_utils import (
    MockTensorStub,
    adapt_backend_module,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        from ml_switcheroo_compiler.core.dtype import DType

        self.dtype = DType.Float32
        self.device = "cpu"
        self.data = "mock_data"


def test_vmap_eager(mocker: MockerFixture) -> None:
    """Test function test_vmap_eager.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.eager_evaluator.EagerEvaluator.evaluate", return_value=t)

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    vmapped = vmap(my_func, in_axes=0, out_axes=0)
    out = require_callable(vmapped)(t)
    assert out == t


def test_vmap_tracing(mocker: MockerFixture) -> None:
    """Test function test_vmap_tracing.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph")

    def my_func(x: BackendArg, y: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    vmapped = vmap(my_func, in_axes=(0, 1), out_axes=0)
    with pytest.raises(AttributeError):
        require_callable(vmapped)(t, t)


def test_vmap_not_tracing() -> None:
    """Test function test_vmap_not_tracing."""
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state_mod

    state_mod.global_tracing_state.is_tracing = False

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    with pytest.raises(RuntimeError):
        require_callable(vmap)(my_func)(Tensor(MockTensor().data, TensorConfig((), DType.Float32, "cpu")))


def test_vmap_eager_impl(mocker: MockerFixture) -> None:
    """Test function test_vmap_eager_impl.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))

    def func(x: BackendArg) -> BackendArg:
        """Test function func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `func`.
        """
        return x

    assert func(t) is t


class MockData:
    """Test class MockData."""

    shape = (2, 3)

    def __init__(self, d: BackendArg = "mock") -> None:
        """Test function __init__.

        Args:
            d: Test parameter `d`.
        """
        self.data = d
        self.id = "mock_id"

    def __array__(self) -> BackendArg:
        """Test function __array__.

        Returns:
            BackendArg: Computed output value from `__array__`.
        """
        return np.zeros((2, 3))

    def tolist(self) -> BackendArg:
        """Test function tolist.

        Returns:
            BackendArg: Computed output value from `tolist`.
        """
        return []


def test_vmap_eager_execute() -> None:
    """Test function test_vmap_eager_execute."""
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.ops.vmap import _eager_vmap

    config.eager_mode = True

    class FakeBackend:
        """Test class FakeBackend."""

        def execute_op(self, name: BackendArg, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """

            class R:
                """Test class R."""

                shape = (2, 3)

            return R()

    old = reg.get_active_backend
    pytest.MonkeyPatch().setattr(reg, "get_active_backend", lambda: FakeBackend(), raising=False)
    t = Tensor(MockData(), TensorConfig((2, 3), DType.Float32, "cpu"))

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    res = _eager_vmap(my_func, 0, 0, (t,))
    assert extract_test_shape(res) == (2, 3)
    reg.get_active_backend = old


def test_vmap_resolve() -> None:
    """Test function test_vmap_resolve."""
    from ml_switcheroo_compiler.ops.vmap import _compute_vmap_shape

    class FakeT(MockTensorStub):
        """Test class FakeT."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = ()
            self.dtype = DType.Float32
            self.device = "cpu"

    assert _compute_vmap_shape(FakeT(), None) == ()


def test_vmap_trace() -> None:
    """Test function test_vmap_trace."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.vmap import _trace_vmap

    class DummyGraph:
        """Test class DummyGraph."""

        def add_node(self, node: BackendArg) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    state.global_tracing_state.is_tracing = True
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "active_graph", DummyGraph(), raising=False)
    pytest.MonkeyPatch().setattr(
        state.global_tracing_state,
        "add_node",
        adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("add_node"),
        raising=False,
    )

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    class FakeT(MockTensorStub):
        """Test class FakeT."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = (2,)
            self.dtype = DType.Float32
            self.device = "cpu"
            self.data = "mock"

    t = Tensor(MockData(), TensorConfig((2,), DType.Float32, "cpu"))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph"):
        res = _trace_vmap(my_func, 0, 0, (t,))
        assert extract_test_shape(res) == (2,)


def test_vmap_non_tensor_arg() -> None:
    """Test function test_vmap_non_tensor_arg."""
    from ml_switcheroo_compiler.ops.vmap import _create_vmap_symbolic_args

    res = _create_vmap_symbolic_args(in_axes=0, args=(1, 2))
    assert res == [1, 2]
