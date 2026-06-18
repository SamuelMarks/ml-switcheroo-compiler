"""Backend Registry."""

import importlib
from typing import Literal

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

BackendName = Literal[
    "jax", "torch", "pytorch", "mlx", "keras", "tensorflow", "numpy", "cupy", "dask"
]


class BackendRegistry:
    """Registry for pluggable backends."""

    _registry: dict[BackendName, type[BaseGenerator]] = {}

    _LAZY_MODULES: dict[str, str] = {
        "jax": "ml_switcheroo_compiler.backends.jax",
        "pytorch": "ml_switcheroo_compiler.backends.pytorch",
        "torch": "ml_switcheroo_compiler.backends.pytorch",
        "mlx": "ml_switcheroo_compiler.backends.mlx",
        "keras": "ml_switcheroo_compiler.backends.keras",
        "tensorflow": "ml_switcheroo_compiler.backends.tensorflow",
        "numpy": "ml_switcheroo_compiler.backends.numpy",
        "cupy": "ml_switcheroo_compiler.backends.cupy",
        "dask": "ml_switcheroo_compiler.backends.dask",
    }

    @classmethod
    def register(cls, name: BackendName, backend_class: type[BaseGenerator]) -> None:
        """Register a backend.

        Args:
            name (str): The name parameter for the operation.
            backend_class (type[BaseGenerator]): The backend_class parameter for the operation.
        """
        cls._registry[name] = backend_class

    @classmethod
    def get(cls, name: BackendName) -> type[BaseGenerator]:
        """Get a backend by name.

        Args:
            name (str): The name parameter for the operation.

        Returns:
            type[BaseGenerator]: The evaluated output resulting from this operation.
        """
        if name not in cls._registry and name in cls._LAZY_MODULES:
            try:
                importlib.import_module(cls._LAZY_MODULES[name])
            except ImportError as e:
                print(f"FAILED TO IMPORT {cls._LAZY_MODULES[name]}: {e}")
                pass

        # In case it registered under a different name like 'pytorch' instead of 'torch'
        if name not in cls._registry:
            if name == "torch" and "pytorch" in cls._registry:
                return cls._registry["pytorch"]

        if name not in cls._registry:
            keys = list(cls._registry.keys()) + list(cls._LAZY_MODULES.keys())
            msg = f"Backend '{name}' not found. Available: {keys}"
            raise ValueError(msg)
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> dict[BackendName, type[BaseGenerator]]:
        """Get all registered backends.

        Returns:
            dict[BackendName, type[BaseGenerator]]: The evaluated output.
        """
        # Ensure all lazy modules are loaded if we want *all* backends
        for name, module_path in cls._LAZY_MODULES.items():
            if name not in cls._registry:
                try:
                    importlib.import_module(module_path)
                except ImportError:
                    pass
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
        name (str): The name parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """

    def decorator(cls: type[BaseGenerator]) -> type[BaseGenerator]:
        """Execute decorator.

        Returns:
        Any: The result.
        """
        BackendRegistry.register(name, cls)
        return cls

    return decorator
