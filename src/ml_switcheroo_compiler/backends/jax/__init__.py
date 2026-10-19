"""Jax Code Generator Package."""

import importlib.util
import sys
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

try:
    _has_pkg = importlib.util.find_spec("jax") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'jax' backend requires the 'jax' and 'jaxlib' libraries to be installed.")

from .distributed_collectives import jax_all_gather, jax_all_reduce, jax_broadcast
from .eager import execute_op
from .generator import JAXCodeGenerator

type.__setattr__(JAXCodeGenerator, "execute_op", cast("EagerCallableProtocol", classmethod)(execute_op))
type.__setattr__(JAXCodeGenerator, "all_reduce", staticmethod(jax_all_reduce))
type.__setattr__(JAXCodeGenerator, "all_gather", staticmethod(jax_all_gather))
type.__setattr__(JAXCodeGenerator, "broadcast", staticmethod(jax_broadcast))
