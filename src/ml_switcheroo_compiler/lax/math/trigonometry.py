"""Trigonometry operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch

from typing import Callable, Any


def _make_dispatcher(op_name: str) -> Callable[..., Any]:
    def _dispatcher(*args: object, **kwargs: object) -> object:
        return dispatch("lax", op_name, *args, **kwargs)

    _dispatcher.__name__ = op_name
    _dispatcher.__doc__ = f"Execute {op_name}."
    return _dispatcher


_OPS = (
    "acos_p",
    "acosh_p",
    "asin_p",
    "asinh_p",
    "atan2_p",
    "atan_p",
    "atanh_p",
    "cos_p",
    "cosh_p",
    "sin_p",
    "sinh_p",
    "tan_p",
    "tanh_p",
)
for _op in _OPS:
    globals()[_op] = _make_dispatcher(_op)
