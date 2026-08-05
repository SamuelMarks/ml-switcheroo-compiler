# ruff: noqa: E501
"""Backend utilities."""

from typing import Any, Callable

import jax.ops
import jax.scipy.linalg
import jax.scipy.signal
import jax.scipy.special
import jax.scipy.special as jss
import jax.scipy.stats


def _execute_adaptive_pool_mock(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_adaptive_pool_mock operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import jax.numpy as jnp

    operand = args[0]
    output_size = args[1]
    if hasattr(operand, "shape"):
        s = list(operand.shape)
        if isinstance(output_size, int):
            out_s = [output_size]
            s[-1] = output_size
        else:
            out_s = list(output_size)
            s[-len(output_size) :] = out_s
        axes = tuple(range(-len(out_s), 0))
        return jnp.broadcast_to(jnp.mean(operand, axis=axes, keepdims=True), s)
    return operand


def _execute_accumulate_n(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_accumulate_n operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = res + inputs[i]
    return res


def _execute_binom_cdf(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_binom_cdf operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    k, n, p = args[0], args[1], args[2]
    loc = kwargs.get("loc", 0.0)
    return jss.betainc(n - (k - loc), (k - loc) + 1, 1 - p)


def _execute_bessel_jn(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_bessel_jn operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return jax.scipy.special.bessel_jn(args[1], v=args[0])


def _execute_unsorted_segment_sum(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_sum operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return jax.ops.segment_sum(*args, **kwargs)


def _execute_unsorted_segment_max(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_max operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return jax.ops.segment_max(*args, **kwargs)


def _execute_unsorted_segment_min(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_min operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return jax.ops.segment_min(*args, **kwargs)


def _execute_unsorted_segment_prod(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_prod operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return jax.ops.segment_prod(*args, **kwargs)


def _execute_variance(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_variance operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import jax.numpy as jnp

    kwargs.setdefault("ddof", 0)
    return jnp.var(*args, **kwargs)


def _execute_cast(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cast operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    tensor = args[0]
    dtype = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dt_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
    import jax.numpy as jnp

    if "int4" in dt_str:
        dt = jnp.int8
    elif "bfloat16" in dt_str:
        dt = jnp.bfloat16
    elif "float16" in dt_str:
        dt = jnp.float16
    elif "float8" in dt_str:
        dt = getattr(jnp, "float8_e4m3fn", jnp.float32)
    else:
        dt = getattr(jnp, dt_str, None)
    return tensor.astype(dt)


def _execute_ragged_tensor_to_dense(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_ragged_tensor_to_dense operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import jax.numpy as jnp

    rt = args[0]
    # Check shape/type via duck typing since jnp.ndarray might be masked
    if isinstance(rt, (list, tuple)) and len(rt) > 0 and hasattr(rt[0], "shape"):
        max_len = max(len(x) for x in rt)
        padded = [jnp.pad(x, (0, max_len - len(x))) for x in rt]
        return jnp.stack(padded)
    return rt


_OP_DISPATCH: dict[str, Callable[..., Any]] = {
    "Variance": _execute_variance,
    "Cumprod": lambda *a, **k: __import__("jax").numpy.cumprod(*a, **k),
    "RaggedTensorToDense": _execute_ragged_tensor_to_dense,
    "Cast": _execute_cast,
    "Convolve2d": jax.scipy.signal.convolve2d,
    "Fftconvolve": jax.scipy.signal.fftconvolve,
    "Welch": jax.scipy.signal.welch,
    "Convolve": jax.scipy.signal.convolve,
    "NormPdf": jax.scipy.stats.norm.pdf,
    "NormCdf": jax.scipy.stats.norm.cdf,
    "GammaPdf": jax.scipy.stats.gamma.pdf,
    "GammaCdf": jax.scipy.stats.gamma.cdf,
    "BetaPdf": jax.scipy.stats.beta.pdf,
    "BetaCdf": jax.scipy.stats.beta.cdf,
    "PoissonPmf": jax.scipy.stats.poisson.pmf,
    "PoissonCdf": jax.scipy.stats.poisson.cdf,
    "BinomPmf": jax.scipy.stats.binom.pmf,
    "BinomCdf": _execute_binom_cdf,
    "Erf": jax.scipy.special.erf,
    "SpecialGamma": jax.scipy.special.gamma,
    "BesselJn": _execute_bessel_jn,
    "Digamma": jax.scipy.special.digamma,
    "Polygamma": jax.scipy.special.polygamma,
    "Zeta": jax.scipy.special.zeta,
    "MatrixExponential": jax.scipy.linalg.expm,
    "Polar": jax.scipy.linalg.polar,
    "Schur": jax.scipy.linalg.schur,
    "SegmentSum": jax.ops.segment_sum,
    "SegmentMax": jax.ops.segment_max,
    "SegmentMin": jax.ops.segment_min,
    "SegmentProd": jax.ops.segment_prod,
    "UnsortedSegmentSum": _execute_unsorted_segment_sum,
    "UnsortedSegmentMax": _execute_unsorted_segment_max,
    "UnsortedSegmentMin": _execute_unsorted_segment_min,
    "UnsortedSegmentProd": _execute_unsorted_segment_prod,
    "AccumulateN": _execute_accumulate_n,
    "AddN": _execute_accumulate_n,
    "ActivityRegularization": lambda x, **kwargs: x,
    "AdaptiveAvgPool2D": _execute_adaptive_pool_mock,
    "AdaptiveAvgPool3D": _execute_adaptive_pool_mock,
    "AdaptiveMaxPool2D": _execute_adaptive_pool_mock,
    "AdaptiveMaxPool3D": _execute_adaptive_pool_mock,
    "AdaptiveMaxPool3D_Indices": lambda *args, **kwargs: (_execute_adaptive_pool_mock(*args, **kwargs), _execute_adaptive_pool_mock(*args, **kwargs)),
    "AdaptiveLogSoftmaxWithLoss": lambda input, target, *args, **kwargs: (target, __import__("jax").numpy.zeros((), dtype=target.dtype)),
    "Adjoint": lambda x, **kwargs: __import__("jax").numpy.conj(__import__("jax").numpy.transpose(x)),
    "AllGather": lambda tensor, *args, **kwargs: __import__("jax").numpy.stack([tensor]),
    "AllToAll": lambda tensor, *args, **kwargs: tensor,
    "AlphaDropout": lambda x, **kwargs: __import__("jax").numpy.where(__import__("jax").random.bernoulli(__import__("jax").random.PRNGKey(0), 1.0 - kwargs.get("p", 0.5), x.shape), x, 0.0),
    "AsString": lambda arr, **kwargs: str(arr),
    "Assert": lambda condition, data, summarize=3, **kwargs: None,
    "Assign": lambda ref, value, **kwargs: value,
    "AssignAdd": lambda ref, value, **kwargs: ref + value,
    "AssignSub": lambda ref, value, **kwargs: ref - value,
    "AssignVariable": lambda ref, value, **kwargs: value,
    "AssociativeScan": lambda *args, **kwargs: args[1] if len(args) > 1 and callable(args[0]) else args[0],
    "Atleast1d": __import__("jax").numpy.atleast_1d,
    "Atleast2d": __import__("jax").numpy.atleast_2d,
    "Atleast3d": __import__("jax").numpy.atleast_3d,
    "AxisIndex": lambda *args, **kwargs: __import__("jax").numpy.array(0),
    "BesselI0": jax.scipy.special.i0,
    "BesselI0e": jax.scipy.special.i0e,
    "BesselI1": jax.scipy.special.i1,
    "BesselI1e": jax.scipy.special.i1e,
    "Frombuffer": lambda *args, **kwargs: __import__("jax").numpy.frombuffer(args[0], **kwargs),
    "Fft2": lambda *args, **kwargs: __import__("jax").numpy.fft.fft2(args[0], **kwargs),
    "Fftfreq": lambda *args, **kwargs: __import__("jax").numpy.fft.fftfreq(*args, **kwargs),
    "Fftn": lambda *args, **kwargs: __import__("jax").numpy.fft.fftn(args[0], **kwargs),
    "Fftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.fftn(args[0], **kwargs),
    "Fftshift": lambda *args, **kwargs: __import__("jax").numpy.fft.fftshift(args[0], **kwargs),
    "HardSilu": lambda x: jax.nn.hard_silu(x),
    "HardSwish": lambda x: jax.nn.hard_swish(x),
    "Hfft": lambda *args, **kwargs: __import__("jax").numpy.fft.hfft(args[0], **kwargs),
    "Ifft": lambda *args, **kwargs: __import__("jax").numpy.fft.ifft(args[0], **kwargs),
    "Ifft2": lambda *args, **kwargs: __import__("jax").numpy.fft.ifft2(args[0], **kwargs),
    "Ifftn": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftn(args[0], **kwargs),
    "Ifftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftn(args[0], **kwargs),
    "Ifftshift": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftshift(args[0], **kwargs),
    "Ihfft": lambda *args, **kwargs: __import__("jax").numpy.fft.ihfft(args[0], **kwargs),
    "Irfft": lambda *args, **kwargs: __import__("jax").numpy.fft.irfft(args[0], **kwargs),
    "Irfft2": lambda *args, **kwargs: __import__("jax").numpy.fft.irfft2(args[0], **kwargs),
    "Irfftn": lambda *args, **kwargs: __import__("jax").numpy.fft.irfftn(args[0], **kwargs),
    "Irfftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.irfftn(args[0], **kwargs),
    "LogSoftmax": lambda *args, **kwargs: __import__("jax").nn.log_softmax(args[0], axis=kwargs.get("axis", -1)),
    "Mish": lambda x: jax.nn.mish(x) if hasattr(jax.nn, "mish") else x * jax.numpy.tanh(jax.numpy.log1p(jax.numpy.exp(x))),
    "OneHot": lambda *args, **kwargs: __import__("jax").nn.one_hot(args[0], kwargs.get("depth", args[1] if len(args) > 1 else 1)),
    "Rfft": lambda *args, **kwargs: __import__("jax").numpy.fft.rfft(args[0], **kwargs),
    "Rfft2": lambda *args, **kwargs: __import__("jax").numpy.fft.rfft2(args[0], **kwargs),
    "Rfftfreq": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftfreq(*args, **kwargs),
    "Rfftn": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftn(args[0], **kwargs),
    "Rfftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftn(args[0], **kwargs),
    "Sigmoid": lambda *args, **kwargs: __import__("jax").nn.sigmoid(args[0]),
    "Softmax": lambda *args, **kwargs: __import__("jax").nn.softmax(args[0], axis=kwargs.get("axis", -1)),
    "Squareplus": lambda x: jax.nn.squareplus(x) if hasattr(jax.nn, "squareplus") else 0.5 * (x + jax.numpy.sqrt(x**2 + 4.0)),
}


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Evaluate execute_op operation.

    Args:
        cls (type): The class.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

    Raises:
        BackendNotSupportedError: An exception.
    """
    if op_type in _OP_DISPATCH:
        return _OP_DISPATCH[op_type](*args, **kwargs)
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_func = global_eager_registry.get(op_type)
    if global_func is not None:
        return global_func(jnp, *args, **kwargs)
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    import jax.lax as lax

    # specific name mappings
    if snake == "mul":
        snake = "multiply"
    elif snake == "sub":
        snake = "subtract"
    elif snake == "div":
        snake = "divide"
    func = None
    for mod in [jnp, lax, getattr(jnp, "linalg", None), getattr(jnp, "fft", None)]:
        if mod is not None and hasattr(mod, snake):
            func = getattr(mod, snake)
            break
    if func is not None:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
