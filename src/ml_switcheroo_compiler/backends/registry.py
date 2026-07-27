# ruff: noqa: E501
"""Backend Registry."""

import logging
from typing import Literal

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.config import config


def _load_numpy() -> None:
    import ml_switcheroo_compiler.backends.numpy  # noqa: F401


def _load_pytorch() -> None:
    import ml_switcheroo_compiler.backends.pytorch  # noqa: F401


def _load_jax() -> None:
    import ml_switcheroo_compiler.backends.jax  # noqa: F401


def _load_tensorflow() -> None:
    import ml_switcheroo_compiler.backends.tensorflow  # noqa: F401


def _load_mlx() -> None:
    import ml_switcheroo_compiler.backends.mlx  # noqa: F401


def _load_dask() -> None:
    import ml_switcheroo_compiler.backends.dask  # noqa: F401


def _load_keras() -> None:
    import ml_switcheroo_compiler.backends.keras  # noqa: F401


def _load_cupy() -> None:
    import ml_switcheroo_compiler.backends.cupy  # noqa: F401


def _load_pure_python() -> None:
    import ml_switcheroo_compiler.backends.pure_python  # noqa: F401


_LOADERS = {
    "numpy": _load_numpy,
    "pytorch": _load_pytorch,
    "torch": _load_pytorch,
    "jax": _load_jax,
    "tensorflow": _load_tensorflow,
    "mlx": _load_mlx,
    "dask": _load_dask,
    "keras": _load_keras,
    "cupy": _load_cupy,
    "pure_python": _load_pure_python,
}

BackendName = Literal["jax", "torch", "pytorch", "mlx", "keras", "tensorflow", "numpy", "cupy", "dask", "pure_python"]


class BackendRegistry:
    """Registry for pluggable backends."""

    _registry: dict[BackendName, type["BaseGenerator"]] = {}

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
        "pure_python": "ml_switcheroo_compiler.backends.pure_python",
    }

    @classmethod
    def register(cls, name: BackendName, backend_class: type["BaseGenerator"]) -> None:
        """Register a backend.

        Args:
            name (str): The name parameter for the operation.
            backend_class (type['BaseGenerator']): The backend_class parameter for the operation.
        """
        cls._registry[name] = backend_class

    @classmethod
    def _try_load_lazy(cls, name: BackendName) -> None:
        """Evaluate and process the try load lazy operation.

        Args:
            name (BackendName): Required parameter for name.

        Returns:
            Any: The evaluated or processed output.
        """
        if name not in cls._registry and name in cls._LAZY_MODULES:
            try:
                if name in _LOADERS:
                    _LOADERS[name]()
            except ImportError as e:
                logging.error(f"FAILED TO IMPORT {cls._LAZY_MODULES[name]}: {e}")

    @classmethod
    def _resolve_alias(cls, name: BackendName) -> BackendName:
        """Evaluate and process the resolve alias operation.

        Args:
            name (BackendName): Required parameter for name.

        Returns:
            BackendName: The evaluated or processed output.
        """
        if name not in cls._registry and name == "torch" and "pytorch" in cls._registry:
            return "pytorch"
        return name

    @classmethod
    def get(cls, name: BackendName) -> type["BaseGenerator"]:
        """Get a backend by name.

        Args:
            name (str): The name parameter for the operation.

        Returns:
            type['BaseGenerator']: The evaluated output resulting from this operation.
        """
        cls._try_load_lazy(name)
        resolved_name = cls._resolve_alias(name)

        if resolved_name not in cls._registry:
            keys = list(cls._registry.keys()) + list(cls._LAZY_MODULES.keys())
            msg = f"Backend '{name}' not found. Available: {keys}"
            raise ValueError(msg)
        return cls._registry[resolved_name]

    @classmethod
    def get_all(cls) -> dict[BackendName, type["BaseGenerator"]]:
        """Get all registered backends.

        Returns:
            dict[BackendName, type['BaseGenerator']]: The evaluated output.
        """
        for name in cls._LAZY_MODULES:
            if name not in cls._registry:
                try:
                    if name in _LOADERS:
                        _LOADERS[name]()
                except ImportError:
                    pass
        return cls._registry.copy()


def get_active_backend() -> type["BaseGenerator"]:
    """Get the currently active backend based on config.

    Returns:
        type['BaseGenerator']: The active backend class.
    """
    return BackendRegistry.get(config.backend)


def register_backend(name: BackendName) -> object:
    """Decorator to register a backend.

    Args:
        name (str): The name parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """

    def decorator(cls: type["BaseGenerator"]) -> type["BaseGenerator"]:
        """Execute decorator.

        Returns:
        Any: The result.
        """
        BackendRegistry.register(name, cls)
        return cls

    return decorator
