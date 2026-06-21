"""Lax primitives."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def abs_p(*args: object, **kwargs: object) -> object:
    """Execute abs_p."""
    return dispatch("lax", "abs_p", *args, **kwargs)


def acos_p(*args: object, **kwargs: object) -> object:
    """Execute acos_p."""
    return dispatch("lax", "acos_p", *args, **kwargs)


def acosh_p(*args: object, **kwargs: object) -> object:
    """Execute acosh_p."""
    return dispatch("lax", "acosh_p", *args, **kwargs)


def add_p(*args: object, **kwargs: object) -> object:
    """Execute add_p."""
    return dispatch("lax", "add_p", *args, **kwargs)


def and_p(*args: object, **kwargs: object) -> object:
    """Execute and_p."""
    return dispatch("lax", "and_p", *args, **kwargs)


def approx_max_k(*args: object, **kwargs: object) -> object:
    """Execute approx_max_k."""
    return dispatch("lax", "approx_max_k", *args, **kwargs)


def approx_min_k(*args: object, **kwargs: object) -> object:
    """Execute approx_min_k."""
    return dispatch("lax", "approx_min_k", *args, **kwargs)


def approx_top_k_p(*args: object, **kwargs: object) -> object:
    """Execute approx_top_k_p."""
    return dispatch("lax", "approx_top_k_p", *args, **kwargs)


def asin_p(*args: object, **kwargs: object) -> object:
    """Execute asin_p."""
    return dispatch("lax", "asin_p", *args, **kwargs)


def asinh_p(*args: object, **kwargs: object) -> object:
    """Execute asinh_p."""
    return dispatch("lax", "asinh_p", *args, **kwargs)


def atan2_p(*args: object, **kwargs: object) -> object:
    """Execute atan2_p."""
    return dispatch("lax", "atan2_p", *args, **kwargs)


def atan_p(*args: object, **kwargs: object) -> object:
    """Execute atan_p."""
    return dispatch("lax", "atan_p", *args, **kwargs)


def atanh_p(*args: object, **kwargs: object) -> object:
    """Execute atanh_p."""
    return dispatch("lax", "atanh_p", *args, **kwargs)


def bessel_i0e(*args: object, **kwargs: object) -> object:
    """Execute bessel_i0e."""
    return dispatch("lax", "bessel_i0e", *args, **kwargs)


def bessel_i0e_p(*args: object, **kwargs: object) -> object:
    """Execute bessel_i0e_p."""
    return dispatch("lax", "bessel_i0e_p", *args, **kwargs)


def bessel_i1e(*args: object, **kwargs: object) -> object:
    """Execute bessel_i1e."""
    return dispatch("lax", "bessel_i1e", *args, **kwargs)


def bessel_i1e_p(*args: object, **kwargs: object) -> object:
    """Execute bessel_i1e_p."""
    return dispatch("lax", "bessel_i1e_p", *args, **kwargs)


def betainc(*args: object, **kwargs: object) -> object:
    """Execute betainc."""
    return dispatch("lax", "betainc", *args, **kwargs)


def cbrt_p(*args: object, **kwargs: object) -> object:
    """Execute cbrt_p."""
    return dispatch("lax", "cbrt_p", *args, **kwargs)


def ceil_p(*args: object, **kwargs: object) -> object:
    """Execute ceil_p."""
    return dispatch("lax", "ceil_p", *args, **kwargs)


def clamp_p(*args: object, **kwargs: object) -> object:
    """Execute clamp_p."""
    return dispatch("lax", "clamp_p", *args, **kwargs)


def clz(*args: object, **kwargs: object) -> object:
    """Execute clz."""
    return dispatch("lax", "clz", *args, **kwargs)


def clz_p(*args: object, **kwargs: object) -> object:
    """Execute clz_p."""
    return dispatch("lax", "clz_p", *args, **kwargs)


def complex(*args: object, **kwargs: object) -> object:
    """Execute complex."""
    return dispatch("lax", "complex", *args, **kwargs)


def complex_p(*args: object, **kwargs: object) -> object:
    """Execute complex_p."""
    return dispatch("lax", "complex_p", *args, **kwargs)


def conj_p(*args: object, **kwargs: object) -> object:
    """Execute conj_p."""
    return dispatch("lax", "conj_p", *args, **kwargs)


def cos_p(*args: object, **kwargs: object) -> object:
    """Execute cos_p."""
    return dispatch("lax", "cos_p", *args, **kwargs)


def cosh_p(*args: object, **kwargs: object) -> object:
    """Execute cosh_p."""
    return dispatch("lax", "cosh_p", *args, **kwargs)


def cumlogsumexp(*args: object, **kwargs: object) -> object:
    """Execute cumlogsumexp."""
    return dispatch("lax", "cumlogsumexp", *args, **kwargs)


def cumlogsumexp_p(*args: object, **kwargs: object) -> object:
    """Execute cumlogsumexp_p."""
    return dispatch("lax", "cumlogsumexp_p", *args, **kwargs)


def cummax(*args: object, **kwargs: object) -> object:
    """Execute cummax."""
    return dispatch("lax", "cummax", *args, **kwargs)


def cummax_p(*args: object, **kwargs: object) -> object:
    """Execute cummax_p."""
    return dispatch("lax", "cummax_p", *args, **kwargs)


def cummin(*args: object, **kwargs: object) -> object:
    """Execute cummin."""
    return dispatch("lax", "cummin", *args, **kwargs)


def cummin_p(*args: object, **kwargs: object) -> object:
    """Execute cummin_p."""
    return dispatch("lax", "cummin_p", *args, **kwargs)


def cumprod(*args: object, **kwargs: object) -> object:
    """Execute cumprod."""
    return dispatch("lax", "cumprod", *args, **kwargs)


def cumprod_p(*args: object, **kwargs: object) -> object:
    """Execute cumprod_p."""
    return dispatch("lax", "cumprod_p", *args, **kwargs)


def cumsum_p(*args: object, **kwargs: object) -> object:
    """Execute cumsum_p."""
    return dispatch("lax", "cumsum_p", *args, **kwargs)


def digamma_p(*args: object, **kwargs: object) -> object:
    """Execute digamma_p."""
    return dispatch("lax", "digamma_p", *args, **kwargs)


def div_p(*args: object, **kwargs: object) -> object:
    """Execute div_p."""
    return dispatch("lax", "div_p", *args, **kwargs)


def eq(*args: object, **kwargs: object) -> object:
    """Execute eq."""
    return dispatch("lax", "eq", *args, **kwargs)


def eq_p(*args: object, **kwargs: object) -> object:
    """Execute eq_p."""
    return dispatch("lax", "eq_p", *args, **kwargs)


def eq_to_p(*args: object, **kwargs: object) -> object:
    """Execute eq_to_p."""
    return dispatch("lax", "eq_to_p", *args, **kwargs)


def erf_inv(*args: object, **kwargs: object) -> object:
    """Execute erf_inv."""
    return dispatch("lax", "erf_inv", *args, **kwargs)


def erf_inv_p(*args: object, **kwargs: object) -> object:
    """Execute erf_inv_p."""
    return dispatch("lax", "erf_inv_p", *args, **kwargs)


def erf_p(*args: object, **kwargs: object) -> object:
    """Execute erf_p."""
    return dispatch("lax", "erf_p", *args, **kwargs)


def erfc_p(*args: object, **kwargs: object) -> object:
    """Execute erfc_p."""
    return dispatch("lax", "erfc_p", *args, **kwargs)


def exp2_p(*args: object, **kwargs: object) -> object:
    """Execute exp2_p."""
    return dispatch("lax", "exp2_p", *args, **kwargs)


def exp_p(*args: object, **kwargs: object) -> object:
    """Execute exp_p."""
    return dispatch("lax", "exp_p", *args, **kwargs)


def expm1_p(*args: object, **kwargs: object) -> object:
    """Execute expm1_p."""
    return dispatch("lax", "expm1_p", *args, **kwargs)


def floor_p(*args: object, **kwargs: object) -> object:
    """Execute floor_p."""
    return dispatch("lax", "floor_p", *args, **kwargs)


def ge(*args: object, **kwargs: object) -> object:
    """Execute ge."""
    return dispatch("lax", "ge", *args, **kwargs)


def ge_p(*args: object, **kwargs: object) -> object:
    """Execute ge_p."""
    return dispatch("lax", "ge_p", *args, **kwargs)


def gt(*args: object, **kwargs: object) -> object:
    """Execute gt."""
    return dispatch("lax", "gt", *args, **kwargs)


def gt_p(*args: object, **kwargs: object) -> object:
    """Execute gt_p."""
    return dispatch("lax", "gt_p", *args, **kwargs)


def igamma(*args: object, **kwargs: object) -> object:
    """Execute igamma."""
    return dispatch("lax", "igamma", *args, **kwargs)


def igamma_grad_a(*args: object, **kwargs: object) -> object:
    """Execute igamma_grad_a."""
    return dispatch("lax", "igamma_grad_a", *args, **kwargs)


def igamma_grad_a_p(*args: object, **kwargs: object) -> object:
    """Execute igamma_grad_a_p."""
    return dispatch("lax", "igamma_grad_a_p", *args, **kwargs)


def igamma_p(*args: object, **kwargs: object) -> object:
    """Execute igamma_p."""
    return dispatch("lax", "igamma_p", *args, **kwargs)


def igammac(*args: object, **kwargs: object) -> object:
    """Execute igammac."""
    return dispatch("lax", "igammac", *args, **kwargs)


def igammac_p(*args: object, **kwargs: object) -> object:
    """Execute igammac_p."""
    return dispatch("lax", "igammac_p", *args, **kwargs)


def imag_p(*args: object, **kwargs: object) -> object:
    """Execute imag_p."""
    return dispatch("lax", "imag_p", *args, **kwargs)


def integer_pow(*args: object, **kwargs: object) -> object:
    """Execute integer_pow."""
    return dispatch("lax", "integer_pow", *args, **kwargs)


def integer_pow_p(*args: object, **kwargs: object) -> object:
    """Execute integer_pow_p."""
    return dispatch("lax", "integer_pow_p", *args, **kwargs)


def is_finite(*args: object, **kwargs: object) -> object:
    """Execute is_finite."""
    return dispatch("lax", "is_finite", *args, **kwargs)


def is_finite_p(*args: object, **kwargs: object) -> object:
    """Execute is_finite_p."""
    return dispatch("lax", "is_finite_p", *args, **kwargs)


def le(*args: object, **kwargs: object) -> object:
    """Execute le."""
    return dispatch("lax", "le", *args, **kwargs)


def le_p(*args: object, **kwargs: object) -> object:
    """Execute le_p."""
    return dispatch("lax", "le_p", *args, **kwargs)


def le_to_p(*args: object, **kwargs: object) -> object:
    """Execute le_to_p."""
    return dispatch("lax", "le_to_p", *args, **kwargs)


def lgamma_p(*args: object, **kwargs: object) -> object:
    """Execute lgamma_p."""
    return dispatch("lax", "lgamma_p", *args, **kwargs)


def log1p_p(*args: object, **kwargs: object) -> object:
    """Execute log1p_p."""
    return dispatch("lax", "log1p_p", *args, **kwargs)


def log_p(*args: object, **kwargs: object) -> object:
    """Execute log_p."""
    return dispatch("lax", "log_p", *args, **kwargs)


def logistic(*args: object, **kwargs: object) -> object:
    """Execute logistic."""
    return dispatch("lax", "logistic", *args, **kwargs)


def logistic_p(*args: object, **kwargs: object) -> object:
    """Execute logistic_p."""
    return dispatch("lax", "logistic_p", *args, **kwargs)


def lt(*args: object, **kwargs: object) -> object:
    """Execute lt."""
    return dispatch("lax", "lt", *args, **kwargs)


def lt_p(*args: object, **kwargs: object) -> object:
    """Execute lt_p."""
    return dispatch("lax", "lt_p", *args, **kwargs)


def lt_to_p(*args: object, **kwargs: object) -> object:
    """Execute lt_to_p."""
    return dispatch("lax", "lt_to_p", *args, **kwargs)


def max_p(*args: object, **kwargs: object) -> object:
    """Execute max_p."""
    return dispatch("lax", "max_p", *args, **kwargs)


def min_p(*args: object, **kwargs: object) -> object:
    """Execute min_p."""
    return dispatch("lax", "min_p", *args, **kwargs)


def mul_p(*args: object, **kwargs: object) -> object:
    """Execute mul_p."""
    return dispatch("lax", "mul_p", *args, **kwargs)


def ne(*args: object, **kwargs: object) -> object:
    """Execute ne."""
    return dispatch("lax", "ne", *args, **kwargs)


def ne_p(*args: object, **kwargs: object) -> object:
    """Execute ne_p."""
    return dispatch("lax", "ne_p", *args, **kwargs)


def neg(*args: object, **kwargs: object) -> object:
    """Execute neg."""
    return dispatch("lax", "neg", *args, **kwargs)


def neg_p(*args: object, **kwargs: object) -> object:
    """Execute neg_p."""
    return dispatch("lax", "neg_p", *args, **kwargs)


def nextafter_p(*args: object, **kwargs: object) -> object:
    """Execute nextafter_p."""
    return dispatch("lax", "nextafter_p", *args, **kwargs)


def not_p(*args: object, **kwargs: object) -> object:
    """Execute not_p."""
    return dispatch("lax", "not_p", *args, **kwargs)


def or_p(*args: object, **kwargs: object) -> object:
    """Execute or_p."""
    return dispatch("lax", "or_p", *args, **kwargs)


def polygamma(*args: object, **kwargs: object) -> object:
    """Execute polygamma."""
    return dispatch("lax", "polygamma", *args, **kwargs)


def polygamma_p(*args: object, **kwargs: object) -> object:
    """Execute polygamma_p."""
    return dispatch("lax", "polygamma_p", *args, **kwargs)


def population_count(*args: object, **kwargs: object) -> object:
    """Execute population_count."""
    return dispatch("lax", "population_count", *args, **kwargs)


def population_count_p(*args: object, **kwargs: object) -> object:
    """Execute population_count_p."""
    return dispatch("lax", "population_count_p", *args, **kwargs)


def pow(*args: object, **kwargs: object) -> object:
    """Execute pow."""
    return dispatch("lax", "pow", *args, **kwargs)


def pow_p(*args: object, **kwargs: object) -> object:
    """Execute pow_p."""
    return dispatch("lax", "pow_p", *args, **kwargs)


def real_p(*args: object, **kwargs: object) -> object:
    """Execute real_p."""
    return dispatch("lax", "real_p", *args, **kwargs)


def reduce_and_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_and_p."""
    return dispatch("lax", "reduce_and_p", *args, **kwargs)


def reduce_max_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_max_p."""
    return dispatch("lax", "reduce_max_p", *args, **kwargs)


def reduce_min_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_min_p."""
    return dispatch("lax", "reduce_min_p", *args, **kwargs)


def reduce_or_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_or_p."""
    return dispatch("lax", "reduce_or_p", *args, **kwargs)


def reduce_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_p."""
    return dispatch("lax", "reduce_p", *args, **kwargs)


def reduce_precision(*args: object, **kwargs: object) -> object:
    """Execute reduce_precision."""
    return dispatch("lax", "reduce_precision", *args, **kwargs)


def reduce_precision_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_precision_p."""
    return dispatch("lax", "reduce_precision_p", *args, **kwargs)


def reduce_prod_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_prod_p."""
    return dispatch("lax", "reduce_prod_p", *args, **kwargs)


def reduce_sum_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_sum_p."""
    return dispatch("lax", "reduce_sum_p", *args, **kwargs)


def reduce_xor_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_xor_p."""
    return dispatch("lax", "reduce_xor_p", *args, **kwargs)


def rem(*args: object, **kwargs: object) -> object:
    """Execute rem."""
    return dispatch("lax", "rem", *args, **kwargs)


def rem_p(*args: object, **kwargs: object) -> object:
    """Execute rem_p."""
    return dispatch("lax", "rem_p", *args, **kwargs)


def rng_bit_generator(*args: object, **kwargs: object) -> object:
    """Execute rng_bit_generator."""
    return dispatch("lax", "rng_bit_generator", *args, **kwargs)


def rng_bit_generator_p(*args: object, **kwargs: object) -> object:
    """Execute rng_bit_generator_p."""
    return dispatch("lax", "rng_bit_generator_p", *args, **kwargs)


def rng_uniform(*args: object, **kwargs: object) -> object:
    """Execute rng_uniform."""
    return dispatch("lax", "rng_uniform", *args, **kwargs)


def rng_uniform_p(*args: object, **kwargs: object) -> object:
    """Execute rng_uniform_p."""
    return dispatch("lax", "rng_uniform_p", *args, **kwargs)


def round_p(*args: object, **kwargs: object) -> object:
    """Execute round_p."""
    return dispatch("lax", "round_p", *args, **kwargs)


def rsqrt_p(*args: object, **kwargs: object) -> object:
    """Execute rsqrt_p."""
    return dispatch("lax", "rsqrt_p", *args, **kwargs)


def scatter_add_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_add_p."""
    return dispatch("lax", "scatter_add_p", *args, **kwargs)


def scatter_max(*args: object, **kwargs: object) -> object:
    """Execute scatter_max."""
    return dispatch("lax", "scatter_max", *args, **kwargs)


def scatter_max_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_max_p."""
    return dispatch("lax", "scatter_max_p", *args, **kwargs)


def scatter_min(*args: object, **kwargs: object) -> object:
    """Execute scatter_min."""
    return dispatch("lax", "scatter_min", *args, **kwargs)


def scatter_min_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_min_p."""
    return dispatch("lax", "scatter_min_p", *args, **kwargs)


def scatter_mul(*args: object, **kwargs: object) -> object:
    """Execute scatter_mul."""
    return dispatch("lax", "scatter_mul", *args, **kwargs)


def scatter_mul_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_mul_p."""
    return dispatch("lax", "scatter_mul_p", *args, **kwargs)


def select_and_scatter_add_p(*args: object, **kwargs: object) -> object:
    """Execute select_and_scatter_add_p."""
    return dispatch("lax", "select_and_scatter_add_p", *args, **kwargs)


def select_and_scatter_p(*args: object, **kwargs: object) -> object:
    """Execute select_and_scatter_p."""
    return dispatch("lax", "select_and_scatter_p", *args, **kwargs)


def shift_left(*args: object, **kwargs: object) -> object:
    """Execute shift_left."""
    return dispatch("lax", "shift_left", *args, **kwargs)


def shift_left_p(*args: object, **kwargs: object) -> object:
    """Execute shift_left_p."""
    return dispatch("lax", "shift_left_p", *args, **kwargs)


def shift_right_arithmetic(*args: object, **kwargs: object) -> object:
    """Execute shift_right_arithmetic."""
    return dispatch("lax", "shift_right_arithmetic", *args, **kwargs)


def shift_right_arithmetic_p(*args: object, **kwargs: object) -> object:
    """Execute shift_right_arithmetic_p."""
    return dispatch("lax", "shift_right_arithmetic_p", *args, **kwargs)


def shift_right_logical(*args: object, **kwargs: object) -> object:
    """Execute shift_right_logical."""
    return dispatch("lax", "shift_right_logical", *args, **kwargs)


def shift_right_logical_p(*args: object, **kwargs: object) -> object:
    """Execute shift_right_logical_p."""
    return dispatch("lax", "shift_right_logical_p", *args, **kwargs)


def sign_p(*args: object, **kwargs: object) -> object:
    """Execute sign_p."""
    return dispatch("lax", "sign_p", *args, **kwargs)


def sin_p(*args: object, **kwargs: object) -> object:
    """Execute sin_p."""
    return dispatch("lax", "sin_p", *args, **kwargs)


def sinh_p(*args: object, **kwargs: object) -> object:
    """Execute sinh_p."""
    return dispatch("lax", "sinh_p", *args, **kwargs)


def sqrt_p(*args: object, **kwargs: object) -> object:
    """Execute sqrt_p."""
    return dispatch("lax", "sqrt_p", *args, **kwargs)


def sub_p(*args: object, **kwargs: object) -> object:
    """Execute sub_p."""
    return dispatch("lax", "sub_p", *args, **kwargs)


def tan_p(*args: object, **kwargs: object) -> object:
    """Execute tan_p."""
    return dispatch("lax", "tan_p", *args, **kwargs)


def tanh_p(*args: object, **kwargs: object) -> object:
    """Execute tanh_p."""
    return dispatch("lax", "tanh_p", *args, **kwargs)


def xor_p(*args: object, **kwargs: object) -> object:
    """Execute xor_p."""
    return dispatch("lax", "xor_p", *args, **kwargs)


def zeta(*args: object, **kwargs: object) -> object:
    """Execute zeta."""
    return dispatch("lax", "zeta", *args, **kwargs)


def zeta_p(*args: object, **kwargs: object) -> object:
    """Execute zeta_p."""
    return dispatch("lax", "zeta_p", *args, **kwargs)
