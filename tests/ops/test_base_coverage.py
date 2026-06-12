"""Defines a mock operation without a dtype for testing purposes.

This module contains the `MockNoDtype` class, which simulates an operation whose numpy
evaluation returns a standard Python integer (which lacks a `dtype` attribute). It also
includes a test case to verify that the framework gracefully handles such outputs by
assigning a default dtype.
"""

from ml_switcheroo.ops.base import OpDef, register_op


@register_op("MockNoDtype")
class MockNoDtype(OpDef):
    """A mock operation that returns a value without a dtype attribute during numpy.

    evaluation

    This class is used to test the framework's fallback behavior when an operation's
    evaluation yields a primitive Python type (like `int`) instead of a NumPy array
    or another object with a `dtype` attribute.
    """

    def numpy_eval(self, *args: object, **kwargs: object) -> int:
        """Evaluates the mock operation using NumPy or standard Python.

        Args:
        *args (object): Positional arguments for the evaluation
        **kwargs (object): Keyword arguments for the evaluation

        Returns:
        int: A constant Python integer (5) which does not possess a `dtype`
        attribute.
        """
        return 5  # Python int has no dtype

    def vjp(self, *args: object, **kwargs: object) -> None:
        """Computes the Vector-Jacobian Product (VJP) for the operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement VJP.
        """

    def jvp(self, *args: object, **kwargs: object) -> None:
        """Computes the Jacobian-Vector Product (JVP) for the operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement JVP.
        """

    def infer_shape(self, *args: object, **kwargs: object) -> None:
        """Infers the output shape of the operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement shape inference.
        """

    def emit_jax(self, *args: object, **kwargs: object) -> None:
        """Emits the JAX translation for this operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement JAX emission.
        """

    def emit_keras(self, *args: object, **kwargs: object) -> None:
        """Emits the Keras translation for this operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement Keras emission.
        """

    def emit_mlx(self, *args: object, **kwargs: object) -> None:
        """Emits the MLX translation for this operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement MLX emission.
        """

    def emit_pytorch(self, *args: object, **kwargs: object) -> None:
        """Emits the PyTorch translation for this operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement PyTorch emission.
        """

    def emit_tensorflow(self, *args: object, **kwargs: object) -> None:
        """Emits the TensorFlow translation for this operation.

        Args:
        *args (object): Positional arguments
        **kwargs (object): Keyword arguments

        Returns:
        None: This mock operation does not implement TensorFlow emission.
        """


def test_base_no_dtype() -> None:
    """Tests that the base operation class handles evaluation outputs lacking a dtype.

    This test configures eager mode, executes the `MockNoDtype` operation,
    and asserts that the resulting output is automatically assigned a default
    dtype of 'float32'

    Returns:
    None
    """
    from ml_switcheroo.core.config import config

    config.eager_mode = True
    op = MockNoDtype()
    res = op(1)
    assert res.dtype.value == "float32"
