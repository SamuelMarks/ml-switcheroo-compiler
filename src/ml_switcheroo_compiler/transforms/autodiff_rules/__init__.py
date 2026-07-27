"""Autodiff rules registry."""

from ml_switcheroo_compiler.transforms.autodiff_rules import (
    binary_division_rules,
    binary_math_rules,
    binary_special_rules,
    binary_trig_rules,
    custom_rules,
    nn_extra_rules,
    reduction_rules,
    shape_creation_rules,
    shape_misc_rules,
    shape_shape_rules,
    signal_rules,
    unary_math_rules,
    unary_misc_rules,
    unary_nn_rules,
)

__all__ = [
    "binary_division_rules",
    "binary_math_rules",
    "binary_special_rules",
    "binary_trig_rules",
    "custom_rules",
    "nn_extra_rules",
    "reduction_rules",
    "shape_creation_rules",
    "shape_misc_rules",
    "shape_shape_rules",
    "signal_rules",
    "unary_math_rules",
    "unary_misc_rules",
    "unary_nn_rules",
]
