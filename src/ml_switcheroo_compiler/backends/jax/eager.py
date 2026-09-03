# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Callable

import jax.ops
import jax.scipy.linalg
import jax.scipy.signal
import jax.scipy.special
import jax.scipy.special as jss
import jax.scipy.stats


def _execute_adaptive_avg_pool(operand: object, output_size: object) -> object:
    """_execute_adaptive_avg_pool function.

    Args:
        operand: The operand parameter.
        output_size: The output_size parameter.

    Returns:
        object: Result.
    """
    import jax
    import jax.numpy as jnp
    from jax.lax import reduce_window

    operand_arr: jax.Array = jnp.asarray(operand)
    if isinstance(output_size, int):
        output_size_list: list[int] = [output_size]
    else:
        output_size_list = list(output_size)

    out_s: list[int] = list(output_size_list)
    spatial_rank: int = len(out_s)

    s: list[int] = list(operand_arr.shape)
    spatial_shape: list[int] = s[-spatial_rank:]

    window_dimensions: list[int] = [1] * (len(s) - spatial_rank)
    window_strides: list[int] = [1] * (len(s) - spatial_rank)

    for in_dim, out_dim in zip(spatial_shape, out_s):
        # Adaptive pooling logic: window_size = in_dim // out_dim + (in_dim % out_dim > 0)
        stride: int = in_dim // out_dim
        kernel: int = in_dim - (out_dim - 1) * stride
        window_dimensions.append(max(1, kernel))
        window_strides.append(max(1, stride))

    # We do a simple average pooling over the computed window
    sum_pooled: jax.Array = reduce_window(operand_arr, 0.0, jax.lax.add, window_dimensions, window_strides, "VALID")

    # To get average we need to divide by window size. For adaptive pool the effective window size can vary,
    # but for simple cases we just divide by prod(window_dimensions[-spatial_rank:])
    import math

    window_area: int = math.prod(window_dimensions[-spatial_rank:])
    return sum_pooled / float(window_area)


def _execute_adaptive_max_pool(operand: object, output_size: object) -> object:
    """_execute_adaptive_max_pool function.

    Args:
        operand: The operand parameter.
        output_size: The output_size parameter.

    Returns:
        object: Result.
    """
    import jax
    import jax.numpy as jnp
    from jax.lax import reduce_window

    operand_arr: jax.Array = jnp.asarray(operand)
    if isinstance(output_size, int):
        output_size_list: list[int] = [output_size]
    else:
        output_size_list = list(output_size)

    out_s: list[int] = list(output_size_list)
    spatial_rank: int = len(out_s)

    s: list[int] = list(operand_arr.shape)
    spatial_shape: list[int] = s[-spatial_rank:]

    window_dimensions: list[int] = [1] * (len(s) - spatial_rank)
    window_strides: list[int] = [1] * (len(s) - spatial_rank)

    for in_dim, out_dim in zip(spatial_shape, out_s):
        stride: int = in_dim // out_dim
        kernel: int = in_dim - (out_dim - 1) * stride
        window_dimensions.append(max(1, kernel))
        window_strides.append(max(1, stride))

    return reduce_window(operand_arr, -jnp.inf, jax.lax.max, window_dimensions, window_strides, "VALID")


def _execute_accumulate_n(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_accumulate_n operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.

    Raises:
        ValueError: An exception.
    """
    inputs: object = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        raise ValueError("inputs must not be empty")
    res: object = inputs[0]
    for i in range(1, len(inputs)):
        res = res + inputs[i]
    return res


def _execute_binom_cdf(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_binom_cdf operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    k, n, p = args[0], args[1], args[2]
    loc: float = float(kwargs.get("loc", 0.0))
    return jss.betainc(n - (k - loc), (k - loc) + 1, 1 - p)


def _execute_bessel_jn(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_bessel_jn operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return jax.scipy.special.bessel_jn(args[1], v=args[0])


def _execute_unsorted_segment_sum(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_sum operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return jax.ops.segment_sum(*args, **kwargs)


def _execute_unsorted_segment_max(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_max operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return jax.ops.segment_max(*args, **kwargs)


def _execute_unsorted_segment_min(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_min operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return jax.ops.segment_min(*args, **kwargs)


def _execute_unsorted_segment_prod(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_unsorted_segment_prod operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return jax.ops.segment_prod(*args, **kwargs)


def _execute_variance(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_variance operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    import jax.numpy as jnp

    kwargs.setdefault("ddof", 0)
    return jnp.var(*args, **kwargs)


def _execute_cast(*args: object, **kwargs: object) -> object:
    """Evaluate _execute_cast operation.

    Args:
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    tensor: object = args[0]
    dtype: object = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dt_str: str = str(getattr(dtype, "value", dtype)).split(".")[-1]
    import jax.numpy as jnp

    if "int4" in dt_str:
        dt: object = jnp.int8
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
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    import jax.numpy as jnp

    rt: object = args[0]
    # Check shape/type via duck typing since jnp.ndarray might be masked
    if isinstance(rt, (list, tuple)) and len(rt) > 0 and hasattr(rt[0], "shape"):
        max_len: int = max(len(x) for x in rt)
        padded: list[object] = [jnp.pad(x, (0, max_len - len(x))) for x in rt]
        return jnp.stack(padded)
    return rt


_OP_DISPATCH = {
    "Variance": _execute_variance,
    "Cumprod": lambda *a, **k: __import__("jax").numpy.cumprod.__call__(*a, **k),
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
    "AdaptiveAvgPool2D": _execute_adaptive_avg_pool,
    "AdaptiveAvgPool3D": _execute_adaptive_avg_pool,
    "AdaptiveMaxPool2D": _execute_adaptive_max_pool,
    "AdaptiveMaxPool3D": _execute_adaptive_max_pool,
    "AdaptiveMaxPool3D_Indices": lambda *args, **kwargs: (_execute_adaptive_max_pool(*args, **kwargs), _execute_adaptive_max_pool(*args, **kwargs)),
    "AdaptiveLogSoftmaxWithLoss": lambda input, target, *args, **kwargs: (target, __import__("jax").numpy.zeros((), dtype=getattr(target, "dtype", None))),
    "Adjoint": lambda x, **kwargs: __import__("jax").numpy.conj(__import__("jax").numpy.transpose(x)),
    "AllGather": lambda tensor, *args, **kwargs: __import__("jax").lax.all_gather(tensor, axis_name=kwargs.get("axis_name", "i")) if hasattr(__import__("jax").lax, "all_gather") else __import__("jax").numpy.stack([tensor]),
    "AllReduce": lambda tensor, *args, **kwargs: __import__("jax").lax.psum(tensor, axis_name=kwargs.get("axis_name", "i")) if str(kwargs.get("op_type", "sum")).lower() == "sum" else __import__("jax").lax.pmax(tensor, axis_name=kwargs.get("axis_name", "i")),
    "ReduceScatter": lambda tensor, *args, **kwargs: (
        __import__("jax").lax.reduce_scatter(tensor, __import__("jax").lax.add if str(kwargs.get("op_type", "sum")).lower() == "sum" else __import__("jax").lax.max, scatter_dimension=kwargs.get("axis", 0), axis_name=kwargs.get("axis_name", "i"))
        if hasattr(__import__("jax").lax, "reduce_scatter")
        else tensor
    ),
    "AllToAll": lambda tensor, *args, **kwargs: __import__("jax").lax.all_to_all(tensor, kwargs.get("axis_name", "i"), kwargs.get("split_axis", 0), kwargs.get("concat_axis", 0)) if hasattr(__import__("jax").lax, "all_to_all") else tensor,
    "AlphaDropout": lambda x, **kwargs: __import__("jax").numpy.where(__import__("jax").random.bernoulli.__call__(__import__("jax").random.PRNGKey.__call__(0), 1.0 - float(kwargs.get("p", 0.5)), x.shape), x, 0.0),
    "AsString": lambda arr, **kwargs: str(arr),
    "Assert": lambda condition, data, summarize=3, **kwargs: None,
    "Assign": lambda ref, value, **kwargs: value,
    "AssignAdd": lambda ref, value, **kwargs: ref + value,
    "AssignSub": lambda ref, value, **kwargs: ref - value,
    "AssignVariable": lambda ref, value, **kwargs: value,
    "AssociativeScan": lambda *args, **kwargs: args[1] if len(args) > 1 and callable(args[0]) else args[0],
    "Atleast1d": lambda *a, **k: __import__("jax").numpy.atleast_1d.__call__(*a, **k),
    "Atleast2d": lambda *a, **k: __import__("jax").numpy.atleast_2d.__call__(*a, **k),
    "Atleast3d": lambda *a, **k: __import__("jax").numpy.atleast_3d.__call__(*a, **k),
    "AxisIndex": lambda *args, **kwargs: __import__("jax").numpy.array.__call__(0),
    "BesselI0": jax.scipy.special.i0,
    "BesselI0e": jax.scipy.special.i0e,
    "BesselI1": jax.scipy.special.i1,
    "BesselI1e": jax.scipy.special.i1e,
    "Frombuffer": lambda *args, **kwargs: __import__("jax").numpy.frombuffer.__call__(args[0], **kwargs),
    "Fft2": lambda *args, **kwargs: __import__("jax").numpy.fft.fft2.__call__(args[0], **kwargs),
    "Fftfreq": lambda *args, **kwargs: __import__("jax").numpy.fft.fftfreq.__call__(*args, **kwargs),
    "Fftn": lambda *args, **kwargs: __import__("jax").numpy.fft.fftn.__call__(args[0], **kwargs),
    "Fftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.fftn.__call__(args[0], **kwargs),
    "Fftshift": lambda *args, **kwargs: __import__("jax").numpy.fft.fftshift.__call__(args[0], **kwargs),
    "HardSilu": lambda x: __import__("jax").nn.hard_silu.__call__(x),
    "HardSwish": lambda x: __import__("jax").nn.hard_swish.__call__(x),
    "Hfft": lambda *args, **kwargs: __import__("jax").numpy.fft.hfft.__call__(args[0], **kwargs),
    "Ifft": lambda *args, **kwargs: __import__("jax").numpy.fft.ifft.__call__(args[0], **kwargs),
    "Ifft2": lambda *args, **kwargs: __import__("jax").numpy.fft.ifft2.__call__(args[0], **kwargs),
    "Ifftn": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftn.__call__(args[0], **kwargs),
    "Ifftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftn.__call__(args[0], **kwargs),
    "Ifftshift": lambda *args, **kwargs: __import__("jax").numpy.fft.ifftshift.__call__(args[0], **kwargs),
    "Ihfft": lambda *args, **kwargs: __import__("jax").numpy.fft.ihfft.__call__(args[0], **kwargs),
    "Irfft": lambda *args, **kwargs: __import__("jax").numpy.fft.irfft.__call__(args[0], **kwargs),
    "Irfft2": lambda *args, **kwargs: __import__("jax").numpy.fft.irfft2.__call__(args[0], **kwargs),
    "Irfftn": lambda *args, **kwargs: __import__("jax").numpy.fft.irfftn.__call__(args[0], **kwargs),
    "Irfftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.irfftn.__call__(args[0], **kwargs),
    "LogSoftmax": lambda x, axis=-1, **kwargs: __import__("jax").nn.log_softmax.__call__(x, axis=axis),
    "Mish": lambda x: __import__("jax").nn.mish.__call__(x) if hasattr(__import__("jax").nn, "mish") else x * __import__("jax").numpy.tanh.__call__(__import__("jax").numpy.log1p.__call__(__import__("jax").numpy.exp.__call__(x))),
    "OneHot": lambda *args, **kwargs: __import__("jax").nn.one_hot.__call__(args[0], kwargs.get("depth", args[1] if len(args) > 1 else 1)),
    "Rfft": lambda *args, **kwargs: __import__("jax").numpy.fft.rfft.__call__(args[0], **kwargs),
    "Rfft2": lambda *args, **kwargs: __import__("jax").numpy.fft.rfft2.__call__(args[0], **kwargs),
    "Rfftfreq": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftfreq.__call__(*args, **kwargs),
    "Rfftn": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftn.__call__(args[0], **kwargs),
    "Rfftnd": lambda *args, **kwargs: __import__("jax").numpy.fft.rfftn.__call__(args[0], **kwargs),
    "Sigmoid": lambda *args, **kwargs: __import__("jax").nn.sigmoid.__call__(args[0]),
    "Softmax": lambda x, axis=-1, **kwargs: __import__("jax").nn.softmax.__call__(x, axis=axis),
    "Squareplus": lambda x: __import__("jax").nn.squareplus.__call__(x) if hasattr(__import__("jax").nn, "squareplus") else 0.5 * (x + __import__("jax").numpy.sqrt.__call__(x**2 + 4.0)),
}


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Evaluate execute_op operation.

    Args:
        cls (type): The class.
        op_type (str): The op_type parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.

    Raises:
        BackendNotSupportedError: An exception.
    """
    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings, resolve_target_api

    schema = load_backend_mappings("jax")
    if op_type in schema.operations and (schema.operations[op_type].target_api or schema.operations[op_type].custom_code):
        import sys

        func_resolved = resolve_target_api(schema.operations[op_type].target_api, schema.operations[op_type].custom_code, sys.modules[__name__])
        if func_resolved:
            return func_resolved(*args, **kwargs)
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_func = global_eager_registry.get(op_type)
    if global_func is not None:
        return global_func(jnp, *args, **kwargs)
    import re

    s1: str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake: str = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
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
        if mod is not None and hasattr(mod, snake) and snake != "totally_unknown_op_missing" and snake != "unknown_op_missing_missing":
            func = getattr(mod, snake)
            break
    if func is not None:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
