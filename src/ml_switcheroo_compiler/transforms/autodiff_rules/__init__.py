"""Autodiff rules registry."""

from ml_switcheroo_compiler.transforms.autodiff_rules import binary_rules as binary_rules
from ml_switcheroo_compiler.transforms.autodiff_rules import linalg_rules as linalg_rules
from ml_switcheroo_compiler.transforms.autodiff_rules import reduction_rules as reduction_rules
from ml_switcheroo_compiler.transforms.autodiff_rules import shape_rules as shape_rules
from ml_switcheroo_compiler.transforms.autodiff_rules import unary_rules as unary_rules

__all__ = [
    "binary_rules",
    "linalg_rules",
    "reduction_rules",
    "shape_rules",
    "unary_rules",
]
