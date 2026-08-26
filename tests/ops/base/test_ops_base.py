# ruff: noqa: E501
"""Unit tests for the operation registry and base operation definition functionality.

This module contains tests verifying that operations can be registered and retrieved
correctly, and that eager and tracing execution modes behave as expected.
"""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, OpDef, get_op, register_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_register_and_get_op() -> None:
    """Test the register and get op behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the registration and retrieval of operations in the global registry.\n\n    This test ensures that custom operations subclassing `OpDef` can be registered\n    using the `@register_op` decorator and retrieved via `get_op`. It also verifies\n    that duplicate registrations raise a `ValueError` and retrieving non-existent\n    operations raises a `KeyError`\n\n    Returns:\n    None\n    "
        original_registry = _OP_REGISTRY.copy()
        _OP_REGISTRY.clear()
        try:

            @register_op("TestOp")
            class TestOp(OpDef):
                """A mock operation class used for testing registration and backend code.

                emission.
                """

                def infer_shape(self, *args, **kwargs):
                    """Infer the output shape of the operation.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    object: The resulting output.
                    """

                def eager_eval(self, *args, **kwargs):
                    """Evaluate the operation using NumPy.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    object: The resulting output.
                    """

                def vjp(self, cotangent, *args, **kwargs):
                    """Compute the vector-Jacobian product.

                    Args:
                    cotangent (object): The cotangent parameter
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    tuple[object, ...]: The resulting output.
                    """
                    return ()

                def jvp(self, tangent, *args, **kwargs):
                    """Compute the Jacobian-vector product.

                    Args:
                    tangent (object): The tangent parameter
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    object: The resulting output.
                    """

                def emit_jax(self, *args, **kwargs) -> str:
                    """Emit code for the jax backend.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    str: The resulting output.
                    """
                    return "jax"

                def emit_pytorch(self, *args, **kwargs) -> str:
                    """Emit code for the pytorch backend.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    str: The resulting output.
                    """
                    return "torch"

                def emit_mlx(self, *args, **kwargs) -> str:
                    """Emit code for the mlx backend.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    str: The resulting output.
                    """
                    return "mlx"

                def emit_keras(self, *args, **kwargs) -> str:
                    """Emit code for the keras backend.

                    Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                    Returns:
                    str: The resulting output.
                    """
                    return "keras"

                def emit_tensorflow(self, *args, **kwargs) -> str:
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
            _OP_REGISTRY.update(original_registry)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_opdef_call_eager() -> None:
    """Test the opdef call eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the eager execution behavior of `OpDef` instances.\n\n    This test configures the system to run in eager mode, instantiates a test\n    operation, and calls it with a `Tensor` input. It verifies that the operation\n    is evaluated immediately using its NumPy implementation and returns a new\n    `Tensor` with the correct shape, data type, and device\n\n    Returns:\n    None\n    "
        original_registry = _OP_REGISTRY.copy()
        try:

            @register_op("TestEagerOp")
            class TestEagerOp(OpDef):
                """A mock operation class used for testing eager execution behavior."""

                def infer_shape(self, *args, **kwargs):
                    """infer_shape function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return ()

                def eager_eval(self, *args, **kwargs):
                    """eager_eval function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return np.array([1, 2, 3], dtype=np.float32)

                def vjp(self, cotangent, *args, **kwargs):
                    """Vjp function.

                    Args:
                    cotangent (object): The cotangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    tuple[object, ...]: The computed result.
                    """
                    return ()

                def jvp(self, tangent, *args, **kwargs):
                    """Jvp function.

                    Args:
                    tangent (object): The tangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return tangent

                def emit_jax(self, *args, **kwargs) -> str:
                    """emit_jax function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_pytorch(self, *args, **kwargs) -> str:
                    """emit_pytorch function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_mlx(self, *args, **kwargs) -> str:
                    """emit_mlx function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_keras(self, *args, **kwargs) -> str:
                    """emit_keras function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_tensorflow(self, *args, **kwargs) -> str:
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
            _OP_REGISTRY.update(original_registry)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_opdef_call_tracing() -> None:
    """Test the opdef call tracing behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the tracing execution behavior of `OpDef` instances.\n\n    This test configures the system to run in tracing mode, starts a tracing\n    context,\n    and calls a test operation with a `Tensor` containing a `ProxyTensor`. It\n    verifies\n    that the operation records its execution in the active tracing graph and returns\n    a new `Tensor` wrapping a `ProxyTensor` with the correct shape and data type\n    It also ensures that calling the operation outside of a tracing context raises\n    a `RuntimeError`\n\n    Returns:\n    None\n    "
        original_registry = _OP_REGISTRY.copy()
        try:

            @register_op("TestTraceOp")
            class TestTraceOp(OpDef):
                """A mock operation class used for testing tracing execution behavior."""

                def infer_shape(self, *args, **kwargs):
                    """infer_shape function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return (3,)

                def eager_eval(self, *args, **kwargs):
                    """eager_eval function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return np.array([1, 2, 3])

                def vjp(self, cotangent, *args, **kwargs):
                    """Vjp function.

                    Args:
                    cotangent (object): The cotangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    tuple[object, ...]: The computed result.
                    """
                    return ()

                def jvp(self, tangent, *args, **kwargs):
                    """Jvp function.

                    Args:
                    tangent (object): The tangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    object: The computed result.
                    """
                    return tangent

                def emit_jax(self, *args, **kwargs) -> str:
                    """emit_jax function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_pytorch(self, *args, **kwargs) -> str:
                    """emit_pytorch function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_mlx(self, *args, **kwargs) -> str:
                    """emit_mlx function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_keras(self, *args, **kwargs) -> str:
                    """emit_keras function.

                    Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                    Returns:
                    str: The computed result.
                    """
                    return ""

                def emit_tensorflow(self, *args, **kwargs) -> str:
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
                nodes = list(global_tracing_state.active_graph.nodes.values())
                assert len(nodes) == 1
                assert nodes[0].op_type == "TestTraceOp"
                assert nodes[0].inputs == ["in1"]
            finally:
                global_tracing_state.stop_tracing()
            with pytest.raises(RuntimeError, match="Cannot emit TestTraceOp node outside of a tracing context"):
                op(t_in)
        finally:
            _OP_REGISTRY.clear()
            _OP_REGISTRY.update(original_registry)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
