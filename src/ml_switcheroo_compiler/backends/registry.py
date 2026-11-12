"""Backend Registry."""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir import LogicalGraph as IRGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol as EagerCallableProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol as EagerTensorProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue as EagerValue,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ShapeProtocol as ShapeProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_bool as ensure_bool,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_bytes as ensure_bytes,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_callable as ensure_callable,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_eager_list as ensure_eager_list,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_eager_tensor as ensure_eager_tensor,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_float as ensure_float,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_int as ensure_int,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_opt_int as ensure_opt_int,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_opt_int_seq as ensure_opt_int_seq,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_opt_str as ensure_opt_str,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_shape as ensure_shape,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_str as ensure_str,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_tensor_pair as ensure_tensor_pair,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    ensure_tensor_triple as ensure_tensor_triple,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol as BackendModuleProtocol,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendNamespaceProtocol as BackendNamespaceProtocol,
)
from ml_switcheroo_compiler.backends.protocols import (
    ensure_mod as ensure_mod,
)
from ml_switcheroo_compiler.core.config import config


class COOTensor:
    """Protocol stub for sparse coordinate tensor compatibility."""

    shape: tuple[int, ...]
    ndim: int
    nnz: int


def _load_numpy() -> None:
    """Load the NumPy backend."""
    import ml_switcheroo_compiler.backends.numpy.generator  # noqa: F401


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


def _load_numba() -> None:
    """Load the Numba backend."""
    import ml_switcheroo_compiler.backends.numba  # noqa: F401


def _load_sparse() -> None:
    """Load the Sparse backend."""
    import ml_switcheroo_compiler.backends.sparse  # noqa: F401


def _load_dpnp() -> None:
    """Load the Data Parallel NumPy (dpnp) backend."""
    import ml_switcheroo_compiler.backends.dpnp  # noqa: F401


def _load_awkward() -> None:
    """Load the Awkward Array backend."""
    import ml_switcheroo_compiler.backends.awkward  # noqa: F401


def _load_pyarrow_compute() -> None:
    """Load the Apache Arrow Compute backend."""
    import ml_switcheroo_compiler.backends.pyarrow_compute  # noqa: F401


def _load_cuda() -> None:
    """Load the CUDA backend."""
    import ml_switcheroo_compiler.backends.cuda.cuda  # noqa: F401


def _load_rocm() -> None:
    """Load the ROCm backend."""
    import ml_switcheroo_compiler.backends.rocm.rocm  # noqa: F401


def _load_metal() -> None:
    """Load the Apple Metal backend."""
    import ml_switcheroo_compiler.backends.metal.metal  # noqa: F401


def _load_edge_webgl() -> None:
    """Load the WebGL backend."""
    import ml_switcheroo_compiler.backends.edge.webgl  # noqa: F401


def _load_webgpu() -> None:
    """Load the WebGPU backend."""
    import ml_switcheroo_compiler.backends.edge.webgpu  # noqa: F401


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
    "edge": _load_edge_onnx,
    "edge_stablehlo": _load_edge_stablehlo,
    "edge_wgsl": _load_edge_wgsl,
    "edge_wasm": _load_edge_wasm_simd,
    "edge_wasm_simd": _load_edge_wasm_simd,
    "edge_webgl": _load_edge_webgl,
    "webgpu": _load_webgpu,
    "cuda": _load_cuda,
    "rocm": _load_rocm,
    "metal": _load_metal,
    "numba": _load_numba,
    "sparse": _load_sparse,
    "sparse_coo": _load_sparse,
    "dpnp": _load_dpnp,
    "awkward": _load_awkward,
    "pyarrow_compute": _load_pyarrow_compute,
}

BackendName = typing.Literal[
    "jax",
    "torch",
    "pytorch",
    "mlx",
    "keras",
    "tensorflow",
    "numpy",
    "cupy",
    "dask",
    "pure_python",
    "llvm_cpp",
    "edge",
    "edge_onnx",
    "edge_stablehlo",
    "edge_wgsl",
    "edge_wasm",
    "edge_wasm_simd",
    "edge_webgl",
    "webgpu",
    "metal",
    "cuda",
    "rocm",
    "numba",
    "sparse",
    "sparse_coo",
    "dpnp",
    "awkward",
    "pyarrow_compute",
]


def _is_backend_name(val: BackendName | str) -> typing.TypeGuard[BackendName]:
    """Type guard narrowing a backend name string to `BackendName`.

    Args:
        val (BackendName | str): Candidate backend name.

    Returns:
        bool: True for valid string names.
    """
    return isinstance(val, str)


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
        "pure_python": "ml_switcheroo_compiler.backends.pure_python",
        "llvm_cpp": "ml_switcheroo_compiler.backends.llvm_cpp",
        "edge": "ml_switcheroo_compiler.backends.edge.onnx",
        "edge_onnx": "ml_switcheroo_compiler.backends.edge.onnx",
        "edge_stablehlo": "ml_switcheroo_compiler.backends.edge.stablehlo",
        "edge_wgsl": "ml_switcheroo_compiler.backends.edge.webgpu",
        "edge_wasm": "ml_switcheroo_compiler.backends.edge.wasm",
        "edge_wasm_simd": "ml_switcheroo_compiler.backends.edge.wasm",
        "edge_webgl": "ml_switcheroo_compiler.backends.edge.webgl",
        "webgpu": "ml_switcheroo_compiler.backends.edge.webgpu",
        "cuda": "ml_switcheroo_compiler.backends.cuda.cuda",
        "rocm": "ml_switcheroo_compiler.backends.rocm.rocm",
        "metal": "ml_switcheroo_compiler.backends.metal.metal",
        "numba": "ml_switcheroo_compiler.backends.numba",
        "sparse": "ml_switcheroo_compiler.backends.sparse",
        "sparse_coo": "ml_switcheroo_compiler.backends.sparse",
        "dpnp": "ml_switcheroo_compiler.backends.dpnp",
        "awkward": "ml_switcheroo_compiler.backends.awkward",
        "pyarrow_compute": "ml_switcheroo_compiler.backends.pyarrow_compute",
    }

    @classmethod
    def register(cls, name: BackendName, backend_class: type[BaseGenerator]) -> None:
        """Register a new backend for compilation and execution.

        Args:
            name (BackendName): The name of the backend (e.g., 'numpy', 'pytorch').
            backend_class (type['BaseGenerator']): The class implementing the backend logic.
        """
        cls._registry[name] = backend_class

    @classmethod
    def _try_load_lazy(cls, name: BackendName | str) -> None:
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
    def _resolve_alias(cls, name: BackendName | str) -> BackendName:
        """Resolve backend name aliases to their canonical names.

        Args:
            name (BackendName): The requested backend name or alias.

        Returns:
            BackendName: The canonical backend name (e.g., 'pytorch' instead of 'torch').
        """
        if name not in cls._registry and name == "torch" and "pytorch" in cls._registry:
            return "pytorch"
        assert _is_backend_name(name)
        return name

    @classmethod
    def get(cls, name: BackendName | str) -> type[BaseGenerator]:
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
    def get_all(cls) -> dict[BackendName, type[BaseGenerator]]:
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
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return cls._registry.copy()

    @classmethod
    def emit_stateful_class(
        cls,
        target_framework: str,
        graph: IRGraph,
        class_name: str = "LoweredModule",
    ) -> str:
        """Emit a stateful object-oriented class definition for `target_framework`.

        Args:
            target_framework (str): Target backend framework identifier.
            graph (IRGraph): Functional IR graph to emit.
            class_name (str): Name of the generated stateful class.

        Returns:
            str: Generated Python source code for the stateful class.
        """
        from ml_switcheroo_compiler.backends._stateful_emitters import emit_backend_stateful_class

        return emit_backend_stateful_class(target_framework, graph, class_name=class_name)


def get_active_backend() -> type[BaseGenerator]:
    """Retrieve the currently active backend class based on global configuration.

    Returns:
        type['BaseGenerator']: The currently active backend class.
    """
    return BackendRegistry.get(config.backend)


def register_backend(name: BackendName) -> Callable[[type[BaseGenerator]], type[BaseGenerator]]:
    """Decorate to register a class as a backend for a specific name.

    Args:
        name (BackendName): The name to register the backend under.

    Returns:
        Callable: A decorator that registers the backend class.
    """

    def decorator(cls: type[BaseGenerator]) -> type[BaseGenerator]:
        """Register the annotated class in the backend registry.

        Args:
            cls (type['BaseGenerator']): The class to register.

        Returns:
            type['BaseGenerator']: The original class.
        """
        BackendRegistry.register(name, cls)
        return cls

    return decorator
