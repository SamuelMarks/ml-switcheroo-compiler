"""Pure Python execution backend package."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.pure_python.eager import execute_op
from ml_switcheroo_compiler.backends.pure_python.generator import PurePythonGenerator
from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor

__all__ = [
    "PurePythonGenerator",
    "PurePythonTensor",
    "execute_op",
]
