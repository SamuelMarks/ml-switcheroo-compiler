"""Jax Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("jax") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'jax' backend requires the 'jax' and 'jaxlib' libraries to be installed.")

from .distributed_collectives import jax_all_gather, jax_all_reduce, jax_broadcast
from .eager import execute_op
from .generator import JAXCodeGenerator

_all_reduce = JAXCodeGenerator.all_reduce
_all_gather = JAXCodeGenerator.all_gather
_broadcast = JAXCodeGenerator.broadcast

__all__ = [
    "JAXCodeGenerator",
    "execute_op",
    "jax_all_gather",
    "jax_all_reduce",
    "jax_broadcast",
]
