"""Autodiff rules registry."""

from ml_switcheroo_compiler.transforms.autodiff_rules import (
    binary_division_rules,
    binary_math_rules,
    binary_special_rules,
    binary_trig_rules,
    cast_and_conj_rules,
    custom_rules,
    generic_shape_rules,
    reduction_rules,
    shape_creation_rules,
    shape_shape_rules,
    signal_rules,
    time_distributed_rules,
    unary_math_rules,
    unary_nn_rules,
)

__all__ = [
    "binary_division_rules",
    "binary_math_rules",
    "binary_special_rules",
    "binary_trig_rules",
    "custom_rules",
    "time_distributed_rules",
    "reduction_rules",
    "shape_creation_rules",
    "generic_shape_rules",
    "shape_shape_rules",
    "signal_rules",
    "unary_math_rules",
    "cast_and_conj_rules",
    "unary_nn_rules",
]
