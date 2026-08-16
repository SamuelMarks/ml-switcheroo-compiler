# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module registry.py."""

from typing import Any

"""Backend Registry."""

import logging
from typing import Callable, Literal

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.config import config


def _load_numpy() -> None:
    """Load the NumPy backend."""
    import ml_switcheroo_compiler.backends.numpy  # noqa: F401


def _load_pytorch() -> None:
    """Load the PyTorch backend."""
    import ml_switcheroo_compiler.backends.pytorch  # noqa: F401


def _load_jax() -> None:
    """Load the JAX backend."""
    import ml_switcheroo_compiler.backends.jax  # noqa: F401


def _load_tensorflow() -> None:
    """Load the TensorFlow backend."""
    import ml_switcheroo_compiler.backends.tensorflow  # noqa: F401


def _load_mlx() -> None:
    """Load the MLX backend."""
    import ml_switcheroo_compiler.backends.mlx  # noqa: F401


def _load_dask() -> None:
    """Load the Dask backend."""
    import ml_switcheroo_compiler.backends.dask  # noqa: F401


def _load_keras() -> None:
    """Load the Keras backend."""
    import ml_switcheroo_compiler.backends.keras  # noqa: F401


def _load_cupy() -> None:
    """Load the CuPy backend."""
    import ml_switcheroo_compiler.backends.cupy  # noqa: F401


def _load_pure_python() -> None:
    """Load the Pure Python backend."""
    import ml_switcheroo_compiler.backends.pure_python  # noqa: F401


def _load_llvm_cpp() -> None:
    """Load the llvm_cpp backend."""
    import ml_switcheroo_compiler.backends.llvm_cpp  # noqa: F401


def _load_edge_onnx() -> None:
    """Load the edge_onnx backend."""
    import ml_switcheroo_compiler.backends.edge.onnx  # noqa: F401


def _load_edge_stablehlo() -> None:
    """Load the edge_stablehlo backend."""
    import ml_switcheroo_compiler.backends.edge.stablehlo  # noqa: F401


def _load_edge_wgsl() -> None:
    """Load the edge_wgsl backend."""
    import ml_switcheroo_compiler.backends.edge.webgpu  # noqa: F401


def _load_edge_wasm_simd() -> None:
    """Load the edge_wasm_simd backend."""
    import ml_switcheroo_compiler.backends.edge.wasm  # noqa: F401


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
    "llvm_cpp": _load_llvm_cpp,
    "edge_onnx": _load_edge_onnx,
    "edge_stablehlo": _load_edge_stablehlo,
    "edge_wgsl": _load_edge_wgsl,
    "edge_wasm_simd": _load_edge_wasm_simd,
}

BackendName = Literal["jax", "torch", "pytorch", "mlx", "keras", "tensorflow", "numpy", "cupy", "dask", "pure_python", "llvm_cpp", "edge_onnx", "edge_stablehlo", "edge_wgsl", "edge_wasm_simd"]


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
        "llvm_cpp": "ml_switcheroo_compiler.backends.llvm_cpp",
        "edge_onnx": "ml_switcheroo_compiler.backends.edge.onnx",
        "edge_stablehlo": "ml_switcheroo_compiler.backends.edge.stablehlo",
        "edge_wgsl": "ml_switcheroo_compiler.backends.edge.webgpu",
        "edge_wasm_simd": "ml_switcheroo_compiler.backends.edge.wasm",
    }

    @classmethod
    def register(cls, name: BackendName, backend_class: type["BaseGenerator"]) -> None:
        """Register a new backend for compilation and execution.

        Args:
            name (BackendName): The name of the backend (e.g., 'numpy', 'pytorch').
            backend_class (type['BaseGenerator']): The class implementing the backend logic.
        """
        cls._registry[name] = backend_class

    @classmethod
    def _try_load_lazy(cls, name: BackendName) -> None:
        """Attempt to lazily load a backend module if it hasn't been loaded yet.

        Args:
            name (BackendName): The name of the backend to load.
        """
        if name not in cls._registry and name in cls._LAZY_MODULES:
            try:
                if name in _LOADERS:
                    _LOADERS[name]()
            except ImportError as e:
                logging.error(f"FAILED TO IMPORT {cls._LAZY_MODULES[name]}: {e}")

    @classmethod
    def _resolve_alias(cls, name: BackendName) -> BackendName:
        """Resolve backend name aliases to their canonical names.

        Args:
            name (BackendName): The requested backend name or alias.

        Returns:
            BackendName: The canonical backend name (e.g., 'pytorch' instead of 'torch').
        """
        if name not in cls._registry and name == "torch" and "pytorch" in cls._registry:
            return "pytorch"
        return name

    @classmethod
    def get(cls, name: BackendName) -> type["BaseGenerator"]:
        """Retrieve a registered backend class by name.

        Args:
            name (BackendName): The name of the backend to retrieve.

        Returns:
            type['BaseGenerator']: The registered backend class.

        Raises:
            ValueError: If the specified backend is not found or cannot be loaded.
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
        """Retrieve all registered backends, loading them if necessary.

        Returns:
            dict[BackendName, type['BaseGenerator']]: A dictionary mapping canonical backend names to their implementing classes.
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
    """Retrieve the currently active backend class based on global configuration.

    Returns:
        type['BaseGenerator']: The currently active backend class.
    """
    return BackendRegistry.get(config.backend)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def register_backend(name: BackendName) -> Callable[[type["BaseGenerator"]], type["BaseGenerator"]]:
    """Decorate to register a class as a backend for a specific name.

    Args:
        name (BackendName): The name to register the backend under.

    Returns:
        Callable: A decorator that registers the backend class.
    """

    def decorator(cls: type["BaseGenerator"]) -> type["BaseGenerator"]:
        """Register the annotated class in the backend registry.

        Args:
            cls (type['BaseGenerator']): The class to register.

        Returns:
            type['BaseGenerator']: The original class.
        """
        BackendRegistry.register(name, cls)
        return cls

    return decorator
