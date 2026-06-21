"""Eager backend registry."""

import typing
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

    def get(self, op_type: str) -> typing.Optional[typing.Callable[..., object]]:
        """Get an eager operation.

        Args:
            op_type (str): The name of the operation.

        Returns:
            typing.Optional[typing.Callable[..., object]]: The eager function or None.
        """
        return self._registry.get(op_type)

    def dispatch(self, op_type: str, *args: object, **kwargs: object) -> object:
        """Dispatch an eager operation.

        Args:
            op_type (str): The name of the operation.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The result.
        """
        func = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)
        msg = f"Operation '{op_type}' not found in registry."
        raise NotImplementedError(msg)


# Global registry instance
global_eager_registry = EagerOpRegistry()

mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
