"""Unit tests for the operation registry and base operation definition functionality.

This module contains tests verifying that operations can be registered and retrieved
correctly, and that eager and tracing execution modes behave as expected.
"""

import pytest

from ml_switcheroo_compiler.ops.base import _OP_REGISTRY, OpDef, get_op, register_op


def test_register_and_get_op() -> None:
    """Tests the registration and retrieval of operations in the global registry.

    This test ensures that custom operations subclassing `OpDef` can be registered
    using the `@register_op` decorator and retrieved via `get_op`. It also verifies
    that duplicate registrations raise a `ValueError` and retrieving non-existent
    operations raises a `KeyError`

    Returns:
    None
    """
    # Clear registry for clean test
    original_registry = _OP_REGISTRY.copy()
    _OP_REGISTRY.clear()

    try:

        @register_op("TestOp")
        class TestOp(OpDef):
            """A mock operation class used for testing registration and backend code.

            emission.
            """

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """Infer the output shape of the operation.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    object: The resulting output.
                """

            def eager_eval(self, *args: object, **kwargs: object) -> object:
                """Evaluate the operation using NumPy.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    object: The resulting output.
                """

            def vjp(
                self,
                cotangent: object,
                *args: object,
                **kwargs: object,
            ) -> tuple[object, ...]:
                """Compute the vector-Jacobian product.

                Args:
                    cotangent (object): The cotangent parameter
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    tuple[object, ...]: The resulting output.
                """
                return ()

            def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
                """Compute the Jacobian-vector product.

                Args:
                    tangent (object): The tangent parameter
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    object: The resulting output.
                """

            def emit_jax(self, *args: object, **kwargs: object) -> str:
                """Emit code for the jax backend.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    str: The resulting output.
                """
                return "jax"

            def emit_pytorch(self, *args: object, **kwargs: object) -> str:
                """Emit code for the pytorch backend.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    str: The resulting output.
                """
                return "torch"

            def emit_mlx(self, *args: object, **kwargs: object) -> str:
                """Emit code for the mlx backend.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    str: The resulting output.
                """
                return "mlx"

            def emit_keras(self, *args: object, **kwargs: object) -> str:
                """Emit code for the keras backend.

                Args:
                    *args (object): Additional keyword arguments.
                    **kwargs (object): Additional keyword arguments.

                Returns:
                    str: The resulting output.
                """
                return "keras"

            def emit_tensorflow(self, *args: object, **kwargs: object) -> str:
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

        with pytest.raises(ValueError, match="already registered"):

            @register_op("TestOp")
            class DuplicateOp(OpDef):
                """A mock operation class used to test duplicate registration prevention."""

        with pytest.raises(KeyError, match="not found in registry"):
            get_op("UnknownOp")

    finally:
        # Restore registry
        _OP_REGISTRY.clear()
        _OP_REGISTRY.update(original_registry)


def test_opdef_call_eager() -> None:
    """Tests the eager execution behavior of `OpDef` instances.

    This test configures the system to run in eager mode, instantiates a test
    operation, and calls it with a `Tensor` input. It verifies that the operation
    is evaluated immediately using its NumPy implementation and returns a new
    `Tensor` with the correct shape, data type, and device

    Returns:
    None
    """
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor

    original_registry = _OP_REGISTRY.copy()
    try:

        @register_op("TestEagerOp")
        class TestEagerOp(OpDef):
            """A mock operation class used for testing eager execution behavior."""

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """infer_shape function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return ()

            def eager_eval(self, *args: object, **kwargs: object) -> object:
                """eager_eval function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return np.array([1, 2, 3], dtype=np.float32)

            def vjp(
                self,
                cotangent: object,
                *args: object,
                **kwargs: object,
            ) -> tuple[object, ...]:
                """Vjp function.

                Args:
                    cotangent (object): The cotangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    tuple[object, ...]: The computed result.
                """
                return ()

            def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
                """Jvp function.

                Args:
                    tangent (object): The tangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return tangent

            def emit_jax(self, *args: object, **kwargs: object) -> str:
                """emit_jax function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_pytorch(self, *args: object, **kwargs: object) -> str:
                """emit_pytorch function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_mlx(self, *args: object, **kwargs: object) -> str:
                """emit_mlx function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_keras(self, *args: object, **kwargs: object) -> str:
                """emit_keras function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_tensorflow(self, *args: object, **kwargs: object) -> str:
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
        t_in = Tensor(data=np.array([0]), shape=(1,), dtype=DType.Float32, device=dev)
        t_out = op(t_in)
        assert isinstance(t_out, Tensor)
        assert t_out.device is dev
        assert t_out.dtype == DType.Float32
        assert list(t_out.shape) == [3]
    finally:
        config.eager_mode = False
        _OP_REGISTRY.clear()
        _OP_REGISTRY.update(original_registry)


def test_opdef_call_tracing() -> None:
    """Tests the tracing execution behavior of `OpDef` instances.

    This test configures the system to run in tracing mode, starts a tracing
    context,
    and calls a test operation with a `Tensor` containing a `ProxyTensor`. It
    verifies
    that the operation records its execution in the active tracing graph and returns
    a new `Tensor` wrapping a `ProxyTensor` with the correct shape and data type
    It also ensures that calling the operation outside of a tracing context raises
    a `RuntimeError`

    Returns:
    None
    """
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    original_registry = _OP_REGISTRY.copy()
    try:

        @register_op("TestTraceOp")
        class TestTraceOp(OpDef):
            """A mock operation class used for testing tracing execution behavior."""

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """infer_shape function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return (3,)

            def eager_eval(self, *args: object, **kwargs: object) -> object:
                """eager_eval function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return np.array([1, 2, 3])

            def vjp(
                self,
                cotangent: object,
                *args: object,
                **kwargs: object,
            ) -> tuple[object, ...]:
                """Vjp function.

                Args:
                    cotangent (object): The cotangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    tuple[object, ...]: The computed result.
                """
                return ()

            def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
                """Jvp function.

                Args:
                    tangent (object): The tangent.
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    object: The computed result.
                """
                return tangent

            def emit_jax(self, *args: object, **kwargs: object) -> str:
                """emit_jax function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_pytorch(self, *args: object, **kwargs: object) -> str:
                """emit_pytorch function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_mlx(self, *args: object, **kwargs: object) -> str:
                """emit_mlx function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_keras(self, *args: object, **kwargs: object) -> str:
                """emit_keras function.

                Args:
                    *args: Additional arguments.
                    **kwargs: Additional keyword arguments.

                Returns:
                    str: The computed result.
                """
                return ""

            def emit_tensorflow(self, *args: object, **kwargs: object) -> str:
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
        t_in = Tensor(data=proxy_in, shape=(1,), dtype=DType.Float32, device=dev)

        _tracer.start_tracing()
        try:
            op = get_op("TestTraceOp")()
            t_out = op(t_in)

            assert isinstance(t_out, Tensor)
            assert isinstance(t_out.data, ProxyTensor)
            assert t_out.device is dev
            assert t_out.shape == (3,)
            assert t_out.dtype == DType.Float32

            # Check node was added
            nodes = list(_tracer.active_graph.nodes.values())
            assert len(nodes) == 1
            assert nodes[0].op_type == "TestTraceOp"
            assert nodes[0].inputs == ["in1"]
        finally:
            _tracer.stop_tracing()
        # Test error when tracing outside context
        with pytest.raises(
            RuntimeError,
            match="Cannot emit TestTraceOp node outside of a tracing context",
        ):
            op(t_in)

    finally:
        _OP_REGISTRY.clear()
        _OP_REGISTRY.update(original_registry)
