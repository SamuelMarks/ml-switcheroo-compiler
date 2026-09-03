# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend code generators for ML Switcheroo Compiler."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend


# Import all backends to register them with the backend registry dynamically
# but gracefully handle missing dependencies by failing registry instead of import
def _safe_import_backend(name: str):
    import importlib

    try:
        importlib.import_module(f"ml_switcheroo_compiler.backends.{name}")
    except ImportError:
        pass


# Force dynamic backend linking imports
_safe_import_backend("numpy")
_safe_import_backend("pytorch")
_safe_import_backend("jax")
_safe_import_backend("mlx")
_safe_import_backend("keras")
_safe_import_backend("tensorflow")
_safe_import_backend("cupy")
_safe_import_backend("dask")
_safe_import_backend("llvm_cpp")
_safe_import_backend("rocm")
_safe_import_backend("metal")
_safe_import_backend("cuda")
_safe_import_backend("edge")

__all__ = [
    "BackendRegistry",
    "BaseGenerator",
    "register_backend",
]
