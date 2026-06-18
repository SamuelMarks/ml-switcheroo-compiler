"""Math Ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Add")
def _np_add(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.add(*args, **kwargs)


@numpy_eager_registry.register("Subtract")
def _np_subtract(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.subtract(*args, **kwargs)


@numpy_eager_registry.register("Multiply")
def _np_multiply(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.multiply(*args, **kwargs)


@numpy_eager_registry.register("TrueDivide")
def _np_true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.divide(*args, **kwargs)


@numpy_eager_registry.register("Exp")
def _np_exp(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.exp(*args, **kwargs)


@numpy_eager_registry.register("Log")
def _np_log(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.log(*args, **kwargs)


@numpy_eager_registry.register("Sin")
def _np_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.sin(*args, **kwargs)


@numpy_eager_registry.register("Cos")
def _np_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.cos(*args, **kwargs)


@numpy_eager_registry.register("Sum")
def _np_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.sum(*args, **kwargs)


@numpy_eager_registry.register("Mean")
def _np_mean(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.mean(*args, **kwargs)


@numpy_eager_registry.register("Max")
def _np_max(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.max(*args, **kwargs)


@numpy_eager_registry.register("Min")
def _np_min(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.min(*args, **kwargs)


@numpy_eager_registry.register("Variance")
def _np_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.var(*args, **kwargs)


@numpy_eager_registry.register("Std")
def _np_std(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.std(*args, **kwargs)


@numpy_eager_registry.register("Argmax")
def _np_argmax(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.argmax(*args, **kwargs)


@numpy_eager_registry.register("Argmin")
def _np_argmin(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.argmin(*args, **kwargs)


@numpy_eager_registry.register("Where")
def _np_where(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.where(*args, **kwargs)


@numpy_eager_registry.register("Prod")
def _np_prod(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.prod(*args, **kwargs)


@numpy_eager_registry.register("All")
def _np_all(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.all(*args, **kwargs)


@numpy_eager_registry.register("AnyOp")
def _np_any_op(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.any(*args, **kwargs)


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.count_nonzero(*args, **kwargs)


@numpy_eager_registry.register("Cumsum")
def _np_cumsum(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.cumsum(*args, **kwargs)


@numpy_eager_registry.register("NanToNum")
def _np_nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.nan_to_num(*args, **kwargs)


@numpy_eager_registry.register("Signbit")
def _np_signbit(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.signbit(*args, **kwargs)


@numpy_eager_registry.register("Frexp")
def _np_frexp(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.frexp(*args, **kwargs)


@numpy_eager_registry.register("Fft")
def _np_fft(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.fft(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.rfft(*args, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.ifft(*args, **kwargs)


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.irfft(*args, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.fftn(*args, **kwargs)


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.ifftn(*args, **kwargs)


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.rfftn(*args, **kwargs)


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.irfftn(*args, **kwargs)


@numpy_eager_registry.register("Fft2")
def _np_fft2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.fft2(*args, **kwargs)


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.ifft2(*args, **kwargs)


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.rfft2(*args, **kwargs)


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.fft.irfft2(*args, **kwargs)


@numpy_eager_registry.register("NotEqual")
def _np_not_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.not_equal(*args, **kwargs)


@numpy_eager_registry.register("Greater")
def _np_greater(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.greater(*args, **kwargs)


@numpy_eager_registry.register("GreaterEqual")
def _np_greater_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.greater_equal(*args, **kwargs)


@numpy_eager_registry.register("Less")
def _np_less(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.less(*args, **kwargs)


@numpy_eager_registry.register("LessEqual")
def _np_less_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.less_equal(*args, **kwargs)


@numpy_eager_registry.register("Maximum")
def _np_maximum(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.maximum(*args, **kwargs)


@numpy_eager_registry.register("Minimum")
def _np_minimum(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.minimum(*args, **kwargs)


@numpy_eager_registry.register("LogicalAnd")
def _np_logical_and(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logical_and(*args, **kwargs)


@numpy_eager_registry.register("LogicalOr")
def _np_logical_or(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logical_or(*args, **kwargs)


@numpy_eager_registry.register("LogicalNot")
def _np_logical_not(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logical_not(*args, **kwargs)


@numpy_eager_registry.register("LogicalXor")
def _np_logical_xor(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logical_xor(*args, **kwargs)


@numpy_eager_registry.register("BitwiseAnd")
def _np_bitwise_and(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.bitwise_and(*args, **kwargs)


@numpy_eager_registry.register("BitwiseOr")
def _np_bitwise_or(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.bitwise_or(*args, **kwargs)


@numpy_eager_registry.register("BitwiseXor")
def _np_bitwise_xor(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.bitwise_xor(*args, **kwargs)


@numpy_eager_registry.register("BitwiseNot")
def _np_bitwise_not(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.bitwise_not(*args, **kwargs)


@numpy_eager_registry.register("LeftShift")
def _np_left_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.left_shift(*args, **kwargs)


@numpy_eager_registry.register("RightShift")
def _np_right_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.right_shift(*args, **kwargs)


@numpy_eager_registry.register("Erf")
def _np_erf(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.erf(*args, **kwargs)


@numpy_eager_registry.register("Erfc")
def _np_erfc(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.erfc(*args, **kwargs)


@numpy_eager_registry.register("Erfinv")
def _np_erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.erfinv(*args, **kwargs)


@numpy_eager_registry.register("Exp2")
def _np_exp2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.exp2(*args, **kwargs)


@numpy_eager_registry.register("Expm1")
def _np_expm1(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.expm1(*args, **kwargs)


@numpy_eager_registry.register("Log1p")
def _np_log1p(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Log2")
def _np_log2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.log2(*args, **kwargs)


@numpy_eager_registry.register("Log10")
def _np_log10(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.log10(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp")
def _np_logaddexp(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logaddexp(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp2")
def _np_logaddexp2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.logaddexp2(*args, **kwargs)


@numpy_eager_registry.register("Round")
def _np_round(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.round(*args, **kwargs)


@numpy_eager_registry.register("Isnan")
def _np_isnan(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.isnan(*args, **kwargs)


@numpy_eager_registry.register("Isinf")
def _np_isinf(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.isinf(*args, **kwargs)


@numpy_eager_registry.register("Isfinite")
def _np_isfinite(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.isfinite(*args, **kwargs)


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.clip(*args, **kwargs)


@numpy_eager_registry.register("Amax")
def _np_amax(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.amax(*args, **kwargs)


@numpy_eager_registry.register("Amin")
def _np_amin(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.amin(*args, **kwargs)


@numpy_eager_registry.register("Logit")
def _np_logit(
    backend_module: object, x: object, eps: object = None, *args: object, **kwargs: object
) -> object:
    return backend_module.log(x / (1.0 - x))


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager import _xlogy

    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager import _mvlgamma

    return _mvlgamma(*args, **kwargs)


@numpy_eager_registry.register("Pmean")
def _np_pmean(
    backend_module: object, x: object, axis_name: object, *args: object, **kwargs: object
) -> object:
    return x


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(
    backend_module: object,
    a: object,
    axis: object = None,
    keepdims: bool = False,
    **kwargs: object,
) -> object:
    import scipy.special

    return scipy.special.logsumexp(a, axis=axis, keepdims=keepdims)


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(
    backend_module: object,
    data: object,
    segment_ids: object,
    num_segments: object = None,
    **kwargs: object,
) -> object:
    import numpy as np

    num_segments = num_segments if num_segments is not None else np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    np.add.at(out, segment_ids, data)
    return out


@numpy_eager_registry.register("Psum")
def _np_psum(backend_module: object, *args: object, **kwargs: object) -> object:
    return args[0]


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Rsqrt")
def _np_rsqrt(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    # Robust rsqrt to avoid division by zero warnings/NaNs in edge cases if required
    # But usually just `np.reciprocal(np.sqrt(x))` or `1.0 / np.sqrt(x)` works.
    # To be extremely robust to 0 or negatives producing uncatchable warnings:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(x)
