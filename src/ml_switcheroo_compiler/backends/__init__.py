# ruff: noqa: E402, E501, F403
"""Backend code generators for ML Switcheroo Compiler."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend


# Import all backends to register them with the backend registry dynamically
# but gracefully handle missing dependencies by failing registry instead of import
def _safe_import_backend(name: str) -> None:
    """Safe import."""
    import importlib

    try:
        importlib.import_module(f"ml_switcheroo_compiler.backends.{name}")
    except Exception:
        pass


# Force dynamic backend linking imports - only foundational numpy is loaded eagerly;
# all other backends are loaded lazily on demand via BackendRegistry._try_load_lazy.
_safe_import_backend("numpy")

__all__ = [
    "BackendRegistry",
    "BaseGenerator",
    "register_backend",
]
