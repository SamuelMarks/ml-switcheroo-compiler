"""Special operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch

from typing import Callable, Any


def _make_dispatcher(op_name: str) -> Callable[..., Any]:
    def _dispatcher(*args: object, **kwargs: object) -> object:
        return dispatch("lax", op_name, *args, **kwargs)

    _dispatcher.__name__ = op_name
    _dispatcher.__doc__ = f"Execute {op_name}."
    return _dispatcher


_OPS = (
    "bessel_i0e",
    "bessel_i0e_p",
    "bessel_i1e",
    "bessel_i1e_p",
    "betainc",
    "digamma_p",
    "erf_inv",
    "erf_inv_p",
    "erf_p",
    "erfc_p",
    "exp2_p",
    "exp_p",
    "expm1_p",
    "igamma",
    "igamma_grad_a",
    "igamma_grad_a_p",
    "igamma_p",
    "igammac",
    "igammac_p",
    "lgamma_p",
    "log1p_p",
    "log_p",
    "logistic",
    "logistic_p",
    "nextafter_p",
    "polygamma",
    "polygamma_p",
    "rsqrt_p",
    "sqrt_p",
    "zeta",
    "zeta_p",
)
for _op in _OPS:
    globals()[_op] = _make_dispatcher(_op)
