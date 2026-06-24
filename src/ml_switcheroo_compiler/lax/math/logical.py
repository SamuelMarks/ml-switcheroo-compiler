"""Logical operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch

from typing import Callable, Any


def _make_dispatcher(op_name: str) -> Callable[..., Any]:
    def _dispatcher(*args: object, **kwargs: object) -> object:
        return dispatch("lax", op_name, *args, **kwargs)

    _dispatcher.__name__ = op_name
    _dispatcher.__doc__ = f"Execute {op_name}."
    return _dispatcher


_OPS = (
    "and_p",
    "clz",
    "clz_p",
    "eq",
    "eq_p",
    "eq_to_p",
    "ge",
    "ge_p",
    "gt",
    "gt_p",
    "is_finite",
    "is_finite_p",
    "le",
    "le_p",
    "le_to_p",
    "lt",
    "lt_p",
    "lt_to_p",
    "ne",
    "ne_p",
    "not_p",
    "or_p",
    "population_count",
    "population_count_p",
    "shift_left",
    "shift_left_p",
    "shift_right_arithmetic",
    "shift_right_arithmetic_p",
    "shift_right_logical",
    "shift_right_logical_p",
    "xor_p",
)
for _op in _OPS:
    globals()[_op] = _make_dispatcher(_op)
