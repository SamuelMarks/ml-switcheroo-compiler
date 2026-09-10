# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
