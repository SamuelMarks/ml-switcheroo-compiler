"""Reductions operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch

from typing import Callable, Any


def _make_dispatcher(op_name: str) -> Callable[..., Any]:
    def _dispatcher(*args: object, **kwargs: object) -> object:
        return dispatch("lax", op_name, *args, **kwargs)

    _dispatcher.__name__ = op_name
    _dispatcher.__doc__ = f"Execute {op_name}."
    return _dispatcher


_OPS = (
    "approx_max_k",
    "approx_min_k",
    "approx_top_k_p",
    "clamp_p",
    "cumlogsumexp",
    "cumlogsumexp_p",
    "cummax",
    "cummax_p",
    "cummin",
    "cummin_p",
    "cumprod",
    "cumprod_p",
    "cumsum_p",
    "max_p",
    "min_p",
    "reduce_and_p",
    "reduce_max_p",
    "reduce_min_p",
    "reduce_or_p",
    "reduce_p",
    "reduce_prod_p",
    "reduce_sum_p",
    "reduce_xor_p",
    "scatter_add_p",
    "scatter_max",
    "scatter_max_p",
    "scatter_min",
    "scatter_min_p",
    "scatter_mul",
    "scatter_mul_p",
    "select_and_scatter_add_p",
    "select_and_scatter_p",
)
for _op in _OPS:
    globals()[_op] = _make_dispatcher(_op)
