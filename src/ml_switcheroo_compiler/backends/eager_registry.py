"""Eager backend registry."""

from collections.abc import Callable


class EagerOpRegistry:
    """Registry for eager operations."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registry: dict[str, Callable] = {}

    def register(self, op_type: str) -> Callable:
        """Register an eager operation.

        Args:
            op_type (str): The operation type.

        Returns:
            Callable: The decorator function.
        """

        def decorator(func: Callable) -> Callable:
            self._registry[op_type] = func
            return func

        return decorator

    def get(self, op_type: str) -> object:
        """Get an eager operation.

        Args:
            op_type (str): The operation type.

        Returns:
            object: The operation function.
        """
        return self._registry.get(op_type)

    def execute(self, op_type: str, *args: object, **kwargs: object) -> object:
        """Execute an eager operation.

        Args:
            op_type (str): The operation type.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The result.
        """
        func = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)  # type: ignore
        msg = f"Operation '{op_type}' not found in registry."
        raise NotImplementedError(msg)


# Global registry instance
global_eager_registry = EagerOpRegistry()

mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
