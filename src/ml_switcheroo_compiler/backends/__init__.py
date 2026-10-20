"""Backend code generators for ML Switcheroo Compiler."""

from __future__ import annotations

import logging

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend


# Import all backends to register them with the backend registry dynamically
# but gracefully handle missing dependencies by failing registry instead of import
def _safe_import_backend(name: str) -> None:
    """Safe import.

    Args:
        name: The name parameter.
    """
    import importlib

    try:
        importlib.import_module(f"ml_switcheroo_compiler.backends.{name}")
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)


# Force dynamic backend linking imports - only foundational numpy is loaded eagerly;
# all other backends are loaded lazily on demand via BackendRegistry._try_load_lazy.
_safe_import_backend("numpy")

__all__ = [
    "BackendRegistry",
    "BaseGenerator",
    "register_backend",
]
