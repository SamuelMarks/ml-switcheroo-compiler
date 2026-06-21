"""Lax primitives."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def associative_scan(*args: object, **kwargs: object) -> object:
    """Execute associative_scan."""
    return dispatch("lax", "associative_scan", *args, **kwargs)


def cond_p(*args: object, **kwargs: object) -> object:
    """Execute cond_p."""
    return dispatch("lax", "cond_p", *args, **kwargs)


def fori_loop(*args: object, **kwargs: object) -> object:
    """Execute fori_loop."""
    return dispatch("lax", "fori_loop", *args, **kwargs)


def map(*args: object, **kwargs: object) -> object:
    """Execute map."""
    return dispatch("lax", "map", *args, **kwargs)


def scan_bind(*args: object, **kwargs: object) -> object:
    """Execute scan_bind."""
    return dispatch("lax", "scan_bind", *args, **kwargs)


def scan_p(*args: object, **kwargs: object) -> object:
    """Execute scan_p."""
    return dispatch("lax", "scan_p", *args, **kwargs)


def switch(*args: object, **kwargs: object) -> object:
    """Execute switch."""
    return dispatch("lax", "switch", *args, **kwargs)


def while_loop(*args: object, **kwargs: object) -> object:
    """Execute while_loop."""
    return dispatch("lax", "while_loop", *args, **kwargs)


def while_p(*args: object, **kwargs: object) -> object:
    """Execute while_p."""
    return dispatch("lax", "while_p", *args, **kwargs)
