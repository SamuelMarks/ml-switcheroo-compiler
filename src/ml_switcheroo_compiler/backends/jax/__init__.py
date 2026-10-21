"""Jax Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable

try:
    _has_pkg = importlib.util.find_spec("jax") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'jax' backend requires the 'jax' and 'jaxlib' libraries to be installed.")

from .distributed_collectives import jax_all_gather, jax_all_reduce, jax_broadcast
from .eager import execute_op
from .generator import JAXCodeGenerator

__all__ = [
    "JAXCodeGenerator",
    "execute_op",
    "jax_all_gather",
    "jax_all_reduce",
    "jax_broadcast",
]


def _all_reduce(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate all_reduce to distributed_collectives.jax_all_reduce.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Reduced tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
    return as_callable(getattr(dc, "jax_all_reduce"))(*args, **kwargs)


def _all_gather(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate all_gather to distributed_collectives.jax_all_gather.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Gathered tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
    return as_callable(getattr(dc, "jax_all_gather"))(*args, **kwargs)


def _broadcast(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate broadcast to distributed_collectives.jax_broadcast.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Broadcasted tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
    return as_callable(getattr(dc, "jax_broadcast"))(*args, **kwargs)


type.__setattr__(JAXCodeGenerator, "execute_op", as_callable(classmethod)(execute_op))
type.__setattr__(JAXCodeGenerator, "all_reduce", staticmethod(_all_reduce))
type.__setattr__(JAXCodeGenerator, "all_gather", staticmethod(_all_gather))
type.__setattr__(JAXCodeGenerator, "broadcast", staticmethod(_broadcast))
