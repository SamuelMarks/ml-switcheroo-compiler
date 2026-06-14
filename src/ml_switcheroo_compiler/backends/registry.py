"""Backend Registry."""

from typing import Literal

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

BackendName = Literal["jax", "torch", "mlx", "keras", "tensorflow", "numpy", "cupy", "dask"]


class BackendRegistry:
    """Registry for pluggable backends."""

    _registry: dict[BackendName, type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: BackendName, backend_class: type[BaseGenerator]) -> None:
        """Register a backend.

        Args:
            name (str): The name.
            backend_class (type[BaseGenerator]): The backend_class.
        """
        cls._registry[name] = backend_class

    @classmethod
    def get(cls, name: BackendName) -> type[BaseGenerator]:
        """Get a backend by name.

        Args:
            name (str): The name.

        Returns:
            type[BaseGenerator]: The computed result.
        """
        if name not in cls._registry:
            keys = list(cls._registry.keys())
            msg = f"Backend '{name}' not found. Available: {keys}"
            raise ValueError(msg)
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> dict[BackendName, type[BaseGenerator]]:
        """Get all registered backends.

        Returns:
            dict[BackendName, type[BaseGenerator]]: The computed result.
        """
        return cls._registry.copy()


def get_active_backend() -> type[BaseGenerator]:
    """Get the currently active backend based on config.

    Returns:
        type[BaseGenerator]: The active backend class.
    """
    from ml_switcheroo_compiler.core.config import config

    return BackendRegistry.get(config.backend)


def register_backend(name: BackendName) -> object:
    """Decorator to register a backend.

    Args:
        name (str): The name.

    Returns:
        object: The computed result.
    """

    def decorator(cls: type[BaseGenerator]) -> type[BaseGenerator]:
        """Execute decorator.

        Returns:
        Any: The result.
        """
        BackendRegistry.register(name, cls)
        return cls

    return decorator
