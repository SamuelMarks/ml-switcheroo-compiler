"""Backend Registry."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


class BackendRegistry:
    """Registry for pluggable backends."""

    _registry: dict[str, type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[BaseGenerator]) -> None:
        """Register a backend."""
        cls._registry[name] = backend_class

    @classmethod
    def get(cls, name: str) -> type[BaseGenerator]:
        """Get a backend by name."""
        if name not in cls._registry:
            keys = list(cls._registry.keys())
            msg = f"Backend '{name}' not found. Available: {keys}"
            raise ValueError(msg)
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> dict[str, type[BaseGenerator]]:
        """Get all registered backends."""
        return cls._registry.copy()


def register_backend(name: str) -> object:
    """Decorator to register a backend."""

    def decorator(cls: type[BaseGenerator]) -> type[BaseGenerator]:
        BackendRegistry.register(name, cls)
        return cls

    return decorator
