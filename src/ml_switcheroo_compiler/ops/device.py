"""Device and system operations."""

from __future__ import annotations


def eval(*args: object) -> None:
    """Forces the evaluation of the given tensors."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "eval"):
        backend.eval(*args)
    else:
        # Fallback for backends that evaluate eagerly
        for arg in args:
            getattr(arg, "data", arg)


def synchronize() -> None:
    """Synchronizes the default device."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "synchronize"):
        backend.synchronize()


def get_peak_memory() -> int:
    """Gets the peak memory usage of the device in bytes."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "get_peak_memory"):
        return backend.get_peak_memory()
    return 0


__all__ = [
    "eval",
    "get_peak_memory",
    "synchronize",
]
