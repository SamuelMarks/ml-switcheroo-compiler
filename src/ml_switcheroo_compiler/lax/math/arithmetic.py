"""Arithmetic operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch

from typing import Callable, Any


def _make_dispatcher(op_name: str) -> Callable[..., Any]:
    def _dispatcher(*args: object, **kwargs: object) -> object:
        return dispatch("lax", op_name, *args, **kwargs)

    _dispatcher.__name__ = op_name
    _dispatcher.__doc__ = f"Execute {op_name}."
    return _dispatcher


_OPS = (
    "abs_p",
    "add_p",
    "cbrt_p",
    "ceil_p",
    "complex",
    "complex_p",
    "conj_p",
    "div_p",
    "floor_p",
    "imag_p",
    "integer_pow",
    "integer_pow_p",
    "mul_p",
    "neg",
    "neg_p",
    "pow",
    "pow_p",
    "real_p",
    "reduce_precision",
    "reduce_precision_p",
    "rem",
    "rem_p",
    "round_p",
    "sign_p",
    "sub_p",
)
for _op in _OPS:
    globals()[_op] = _make_dispatcher(_op)
