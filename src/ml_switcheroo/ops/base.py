"""Base definitions for the operation registry."""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound="OpDef")

# Global operation registry
_OP_REGISTRY: dict[str, type["OpDef"]] = {}


class OpDef(ABC):
    """Abstract base class for all operations in the compiler."""

    @abstractmethod
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape(s) and dtype(s) of the operation.

        Args:
            *args: Positional arguments (typically TensorSpec or shapes).
            **kwargs: Keyword arguments for the operation.

        Returns:
            The output TensorSpec(s) or shape(s).
        """
        ...

    @abstractmethod
    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation eagerly using NumPy.

        Args:
            *args: Positional arguments (NumPy arrays or scalars).
            **kwargs: Keyword arguments.

        Returns:
            The result of the operation as a NumPy array or scalar.
        """
        ...

    @abstractmethod
    def vjp(
        self, cotangent: object, *args: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Compute the Vector-Jacobian Product for reverse-mode autodiff.

        Args:
            cotangent: The cotangent from the upstream operation.
            *args: The primal inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of gradients with respect to each input argument.
        """
        ...

    @abstractmethod
    def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
        """Compute the Jacobian-Vector Product for forward-mode autodiff.

        Args:
            tangent: The tangent of the inputs.
            *args: The primal inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            The tangent of the output.
        """
        ...

    @abstractmethod
    def emit_jax(self, *args: object, **kwargs: object) -> str:
        """Emit the JAX code template for this operation.

        Args:
            *args: Variable names or expressions as strings.
            **kwargs: Additional backend-specific formatting options.

        Returns:
            The JAX code string.
        """
        ...

    @abstractmethod
    def emit_pytorch(self, *args: object, **kwargs: object) -> str:
        """Emit the PyTorch code template for this operation.

        Args:
            *args: Variable names or expressions as strings.
            **kwargs: Additional backend-specific formatting options.

        Returns:
            The PyTorch code string.
        """
        ...

    @abstractmethod
    def emit_mlx(self, *args: object, **kwargs: object) -> str:
        """Emit the MLX code template for this operation.

        Args:
            *args: Variable names or expressions as strings.
            **kwargs: Additional backend-specific formatting options.

        Returns:
            The MLX code string.
        """
        ...

    @abstractmethod
    def emit_keras(self, *args: object, **kwargs: object) -> str:
        """Emit the Keras code template for this operation.

        Args:
            *args: Variable names or expressions as strings.
            **kwargs: Additional backend-specific formatting options.

        Returns:
            The Keras code string.
        """
        ...

    @abstractmethod
    def emit_tensorflow(self, *args: object, **kwargs: object) -> str:
        """Emit the TensorFlow code template for this operation.

        Args:
            *args: Variable names or expressions as strings.
            **kwargs: Additional backend-specific formatting options.

        Returns:
            The TensorFlow code string.
        """
        ...


def register_op(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register an operation class in the global registry.

    Args:
        name: The unique string name of the operation (e.g., 'Add', 'Sin').

    Returns:
        A class decorator.
    """

    def decorator(cls: type[T]) -> type[T]:
        """Docstring."""
        if name in _OP_REGISTRY:
            raise ValueError(f"Operation '{name}' is already registered.")
        _OP_REGISTRY[name] = cls
        return cls

    return decorator


def get_op(name: str) -> type[OpDef]:
    """Retrieve an operation class by name.

    Args:
        name: The name of the operation.

    Returns:
        The operation class.

    Raises:
        KeyError: If the operation is not registered.
    """
    if name not in _OP_REGISTRY:
        raise KeyError(f"Operation '{name}' not found in registry.")
    return _OP_REGISTRY[name]
