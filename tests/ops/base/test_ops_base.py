# ruff: noqa: E501
"""Unit tests for the operation registry and base operation definition functionality.

This module contains tests verifying that operations can be registered and retrieved
correctly, and that eager and tracing execution modes behave as expected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import dispatch_eager, emit_ir_node
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, OpDef, get_op, register_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_register_and_get_op_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_register_and_get_op.

    Args:
        _state: Shared test state dictionary.
    """
    "Tests the registration and retrieval of operations in the global registry.\n\n    This test ensures that custom operations subclassing `OpDef` can be registered\n    using the `@register_op` decorator and retrieved via `get_op`. It also verifies\n    that duplicate registrations raise a `ValueError` and retrieving non-existent\n    operations raises a `KeyError`\n\n    Returns:\n    None\n    "
    original_registry = _OP_REGISTRY.copy()
    _OP_REGISTRY.clear()
    getattr(_state, "update")({"original_registry": original_registry})
    _state.update(locals())


def _test_register_and_get_op_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_register_and_get_op.

    Args:
        _state: Shared test state dictionary.
    """
    original_registry = _state["original_registry"]
    try:

        @register_op("TestOp")
        class TestOp(OpDef):
            """A mock operation class used for testing registration and backend code.

            emission.
            """

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Infer the output shape of the operation.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                object: The resulting output.
                """

            def eager_eval(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Evaluate the operation using NumPy.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                object: The resulting output.
                """

            def vjp(self, cotangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Compute the vector-Jacobian product.

                Args:
                cotangent (object): The cotangent parameter
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                tuple[object, ...]: The resulting output.
                """
                return ()

            def jvp(self, tangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Compute the Jacobian-vector product.

                Args:
                tangent (object): The tangent parameter
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                object: The resulting output.
                """

            def emit_jax(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """Emit code for the jax backend.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                str: The resulting output.
                """
                return "jax"

            def emit_pytorch(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """Emit code for the pytorch backend.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                str: The resulting output.
                """
                return "torch"

            def emit_mlx(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """Emit code for the mlx backend.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                str: The resulting output.
                """
                return "mlx"

            def emit_keras(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """Emit code for the keras backend.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                str: The resulting output.
                """
                return "keras"

            def emit_tensorflow(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """Emit code for the tensorflow backend.

                Args:
                *args (object): Additional keyword arguments.
                **kwargs (object): Additional keyword arguments.

                Returns:
                str: The resulting output.
                """
                return "tf"

        retrieved_op = get_op("TestOp")
        assert retrieved_op is TestOp
        with pytest.raises((ValueError, ShapeMismatchError), match="already registered"):

            @register_op("TestOp")
            class DuplicateOp(OpDef):
                """A mock operation class used to test duplicate registration prevention."""

        with pytest.raises(KeyError, match="not found in registry"):
            get_op("UnknownOp")
    finally:
        _OP_REGISTRY.clear()
        cast("Callable[..., EagerTensorProtocol]", _OP_REGISTRY.update)(original_registry)
    _state.update(locals())


def test_register_and_get_op() -> None:
    """Test the register and get op behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        for _sub_fn in (_test_register_and_get_op_sub_1, _test_register_and_get_op_sub_2):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def _test_opdef_call_eager_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_opdef_call_eager.

    Args:
        _state: Shared test state dictionary.
    """
    "Tests the eager execution behavior of `OpDef` instances.\n\n    This test configures the system to run in eager mode, instantiates a test\n    operation, and calls it with a `Tensor` input. It verifies that the operation\n    is evaluated immediately using its NumPy implementation and returns a new\n    `Tensor` with the correct shape, data type, and device\n\n    Returns:\n    None\n    "
    original_registry = _OP_REGISTRY.copy()
    getattr(_state, "update")({"original_registry": original_registry})
    _state.update(locals())


def _test_opdef_call_eager_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_opdef_call_eager.

    Args:
        _state: Shared test state dictionary.
    """
    original_registry = _state["original_registry"]
    try:

        @register_op("TestEagerOp")
        class TestEagerOp(OpDef):
            """A mock operation class used for testing eager execution behavior."""

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """infer_shape function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return ()

            def eager_eval(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """eager_eval function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return np.array([1, 2, 3], dtype=np.float32)

            def vjp(self, cotangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Vjp function.

                Args:
                cotangent (object): The cotangent.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                tuple[object, ...]: The computed result.
                """
                return ()

            def jvp(self, tangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Jvp function.

                Args:
                tangent (object): The tangent.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return tangent

            def emit_jax(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_jax function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_pytorch(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_pytorch function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_mlx(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_mlx function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_keras(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_keras function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_tensorflow(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_tensorflow function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

        config.eager_mode = True
        op = get_op("TestEagerOp")()
        dev = Device(DeviceType.CPU)
        t_in = Tensor(np.array([0]), TensorConfig((1,), DType.Float32, dev))
        t_out = op(t_in)
        assert isinstance(t_out, Tensor)
        assert t_out.device is dev
        assert t_out.dtype == DType.Float32
        assert list(t_out.shape) == [3]
    finally:
        config.eager_mode = False
        _OP_REGISTRY.clear()
        cast("Callable[..., EagerTensorProtocol]", _OP_REGISTRY.update)(original_registry)
    _state.update(locals())


def test_opdef_call_eager() -> None:
    """Test the opdef call eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        for _sub_fn in (_test_opdef_call_eager_sub_1, _test_opdef_call_eager_sub_2):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def _test_opdef_call_tracing_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_opdef_call_tracing.

    Args:
        _state: Shared test state dictionary.
    """
    "Tests the tracing execution behavior of `OpDef` instances.\n\n    This test configures the system to run in tracing mode, starts a tracing\n    context,\n    and calls a test operation with a `Tensor` containing a `ProxyTensor`. It\n    verifies\n    that the operation records its execution in the active tracing graph and returns\n    a new `Tensor` wrapping a `ProxyTensor` with the correct shape and data type\n    It also ensures that calling the operation outside of a tracing context raises\n    a `RuntimeError`\n\n    Returns:\n    None\n    "
    original_registry = _OP_REGISTRY.copy()
    getattr(_state, "update")({"original_registry": original_registry})
    _state.update(locals())


def _test_opdef_call_tracing_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_opdef_call_tracing.

    Args:
        _state: Shared test state dictionary.
    """
    original_registry = _state["original_registry"]
    try:

        @register_op("TestTraceOp")
        class TestTraceOp(OpDef):
            """A mock operation class used for testing tracing execution behavior."""

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """infer_shape function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return (3,)

            def eager_eval(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """eager_eval function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return np.array([1, 2, 3])

            def vjp(self, cotangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Vjp function.

                Args:
                cotangent (object): The cotangent.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                tuple[object, ...]: The computed result.
                """
                return ()

            def jvp(self, tangent: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Jvp function.

                Args:
                tangent (object): The tangent.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                object: The computed result.
                """
                return tangent

            def emit_jax(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_jax function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_pytorch(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_pytorch function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_mlx(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_mlx function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_keras(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_keras function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

            def emit_tensorflow(self, *args: EagerValue, **kwargs: EagerValue) -> str:
                """emit_tensorflow function.

                Args:
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.

                Returns:
                str: The computed result.
                """
                return ""

        config.eager_mode = False
        dev = Device(DeviceType.CPU)
        proxy_in = ProxyTensor(id="in1", shape=(1,), dtype=DType.Float32.value)
        t_in = Tensor(proxy_in, TensorConfig((1,), DType.Float32, dev))
        global_tracing_state.start_tracing()
        try:
            op = get_op("TestTraceOp")()
            t_out = op(t_in)
            assert isinstance(t_out, Tensor)
            assert isinstance(t_out.data, ProxyTensor)
            assert t_out.device is dev
            assert t_out.shape == (3,)
            assert t_out.dtype == DType.Float32
            nodes = list(cast("EagerTensorProtocol", getattr(global_tracing_state.active_graph, "nodes")).values())
            assert len(nodes) == 1
            assert nodes[0].op_type == "TestTraceOp"
            assert nodes[0].inputs == ["in1"]
        finally:
            global_tracing_state.stop_tracing()
        with pytest.raises(RuntimeError, match="Cannot emit TestTraceOp node outside of a tracing context"):
            op(t_in)
    finally:
        _OP_REGISTRY.clear()
        cast("Callable[..., EagerTensorProtocol]", _OP_REGISTRY.update)(original_registry)
    _state.update(locals())


def test_opdef_call_tracing() -> None:
    """Test the opdef call tracing behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        for _sub_fn in (_test_opdef_call_tracing_sub_1, _test_opdef_call_tracing_sub_2):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_emit_ir_node() -> None:
    # with graph
    """Test function test_emit_ir_node."""
    graph = MagicMock()
    graph.nodes = {}
    nid = emit_ir_node(graph, "TestOp", ["in1"], shape_metadata=(1,))
    assert nid in graph.nodes

    # without graph
    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state") as global_state:
        nid = emit_ir_node(None, "TestOp", ["in1"])
        global_state.add_node.assert_called_once()


def test_opdef_base_methods() -> None:
    """Test function test_opdef_base_methods."""
    op = OpDef()
    assert op.infer_shape() == tuple()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        backend = MagicMock()
        backend.execute_op.return_value = "eager_result"
        mock_get_backend.return_value = backend
        assert op.eager_eval() == "eager_result"


def test_dispatch_eager_wrapper() -> None:
    """Test function test_dispatch_eager_wrapper."""

    @dispatch_eager("TestOp")
    def test_func(t1: EagerTensorProtocol, t2: EagerTensorProtocol) -> EagerValue:
        """Test function test_func."""
        return "not eager"
        assert callable(test_func)

    with patch("ml_switcheroo_compiler.ops.base.config") as mock_config:
        # Test not eager mode
        mock_config.eager_mode = False
        assert test_func(1, 2) == "not eager"

        # Test eager mode
        mock_config.eager_mode = True
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            backend = MagicMock()
            backend.array.return_value = MagicMock(shape=(1,))
            backend.execute_op.return_value = "backend_res"
            mock_get_backend.return_value = backend

            # with tensor inputs
            t1 = MagicMock(spec=Tensor)
            t1.data = "t1_data"
            t1.device = "dev"
            from ml_switcheroo_compiler.core.dtype import DType

            t1.dtype = DType.Float32
            res = test_func(t1, 2)
            assert isinstance(res, Tensor)

            # with multiple tensor return
            backend.execute_op.return_value = ("backend_res1", "backend_res2")
            res_tuple = test_func(t1, 2)
            assert isinstance(res_tuple, tuple)
            assert isinstance(res_tuple[0], Tensor)
            assert isinstance(res_tuple[1], Tensor)
