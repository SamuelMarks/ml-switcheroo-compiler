"""Tensorflow Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable
from ml_switcheroo_compiler.core.shape import StructuralOperandProtocol

try:
    _has_pkg = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'tensorflow' backend requires the 'tensorflow' library to be installed.")

from .eager import execute_op
from .generator import TensorFlowCodeGenerator


class _TFGeneratorClassProtocol(typing.Protocol):
    """Protocol for attaching TensorFlow `execute_op` classmethod."""

    execute_op: EagerValue


def _is_tf_gen_cls(cls_obj: StructuralOperandProtocol) -> typing.TypeGuard[_TFGeneratorClassProtocol]:
    """Type guard narrowing `cls_obj` to `_TFGeneratorClassProtocol`.

    Args:
        cls_obj (StructuralOperandProtocol): Target class.

    Returns:
        bool: Always True.
    """
    return cls_obj is not None


_raw_tf_cls: StructuralOperandProtocol = TensorFlowCodeGenerator
assert _is_tf_gen_cls(_raw_tf_cls)
_raw_tf_cls.execute_op = as_callable(classmethod)(execute_op)
