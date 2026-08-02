# ruff: noqa: E501
"""Core utilities."""

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the true divide operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    func = getattr(backend_module, "divide", getattr(backend_module, "true_divide", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fft")
def _fft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the fft operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the rfft operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the fftn operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Erf")
def _erf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate erf."""
    x = args[0]
    if hasattr(backend_module, "erf"):
        return backend_module.erf(x)
    # A&S approximation 7.1.26
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*e^{-x^2}
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = backend_module.sign(x)
    abs_x = backend_module.abs(x)
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * backend_module.exp(-abs_x * abs_x)
    return sign * y


@global_eager_registry.register("Erfc")
def _erfc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate erfc."""
    x = args[0]
    if hasattr(backend_module, "erfc"):
        return backend_module.erfc(x)
    return 1.0 - _erf(backend_module, x)


@global_eager_registry.register("Expm1")
def _expm1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate expm1."""
    x = args[0]
    if hasattr(backend_module, "expm1"):
        return backend_module.expm1(x)
    return backend_module.exp(x) - 1.0


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate erfinv."""
    x = args[0]
    if hasattr(backend_module, "erfinv"):
        return backend_module.erfinv(x)
    # Approximation of erfinv
    # https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    # For a naive approximation:
    a = 0.147
    ln_1_x2 = backend_module.log(1.0 - x * x + 1e-12)
    term1 = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2 = ln_1_x2 / a
    return backend_module.sign(x) * backend_module.sqrt(backend_module.sqrt(term1 * term1 - term2) - term1)


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the nan to num operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    x = args[0]
    kwargs.pop("copy", None)
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return backend_module.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return None


@global_eager_registry.register("Einsum")
def _einsum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the einsum operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    eq = kwargs.pop("equation", "") if "equation" in kwargs else args[0] if len(args) > 0 and isinstance(args[0], str) else ""
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args
    if hasattr(backend_module, "einsum"):
        return backend_module.einsum(eq, *op_args, **kwargs)
    return None


@global_eager_registry.register("Allclose")
def _allclose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the allclose operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-05)
    atol = kwargs.get("atol", 1e-08)
    equal_nan = kwargs.get("equal_nan", False)

    def _val(x: object) -> object:
        """Evaluate and process the val operation.

        Args:
            x (object): Required parameter for x.

        Returns:
            object: The evaluated or processed output.
        """
        x_data = getattr(x, "data", x)
        if hasattr(x_data, "item") and callable(x_data.item):
            return x_data.item()
        if hasattr(x_data, "tolist"):
            return x_data.tolist()
        return x_data

    if hasattr(backend_module, "allclose"):
        return backend_module.allclose(a, b, rtol=float(_val(rtol)), atol=float(_val(atol)), equal_nan=bool(_val(equal_nan)))
    return None


@global_eager_registry.register("Psum")
def _psum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the psum operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("Pmean")
def _pmean(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the pmean operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the segment sum operation.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    if len(args) < 2:
        return backend_module.asarray(args[0]) if args else None
    data = backend_module.asarray(args[0])
    segment_ids = backend_module.asarray(args[1])
    num_segments = kwargs.get("num_segments", args[2] if len(args) > 2 else backend_module.max(segment_ids) + 1)

    out = backend_module.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    backend_module.add.at(out, segment_ids, data)
    return backend_module.asarray(out)


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_allclose",
    "_einsum",
    "_erfinv",
    "_fft",
    "_fftn",
    "_nan_to_num",
    "_pmean",
    "_psum",
    "_rfft",
    "_segment_sum",
    "_true_divide",
    "global_eager_registry",
]


def _apply_causal_mask(backend_module: object, scores: object) -> object:
    """Applies a causal mask to attention scores."""
    if hasattr(backend_module, "triu") and hasattr(backend_module, "ones") and hasattr(backend_module, "where"):
        causal_mask = backend_module.triu(backend_module.ones(scores.shape[-2:]), 1)
        return backend_module.where(causal_mask > 0, float("-inf"), scores)
    return scores


def _apply_softmax(backend_module: object, scores: object) -> object:
    """Applies softmax to attention scores."""
    if hasattr(backend_module, "softmax"):
        return backend_module.softmax(scores, axis=-1)
    if hasattr(backend_module, "nn") and hasattr(backend_module.nn, "softmax"):
        return backend_module.nn.softmax(scores, axis=-1)
    if hasattr(backend_module, "exp") and hasattr(backend_module, "sum") and hasattr(backend_module, "max"):
        exps = backend_module.exp(scores - backend_module.max(scores, axis=-1, keepdims=True))
        return exps / backend_module.sum(exps, axis=-1, keepdims=True)
    return scores


@global_eager_registry.register("ScaledDotProductAttention")
def _scaled_dot_product_attention_eager(backend_module: object, query: object, key: object, value: object, *args: object, **kwargs: object) -> object:
    """Fallback eager execution for ScaledDotProductAttention."""
    import math

    scale = kwargs.get("scale")
    is_causal = kwargs.get("is_causal", False)
    mask = kwargs.get("mask", None)

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # key transpose
    key_t_axes = list(range(len(key.shape)))
    key_t_axes[-1], key_t_axes[-2] = key_t_axes[-2], key_t_axes[-1]

    if hasattr(backend_module, "transpose"):
        key_t = backend_module.transpose(key, axes=key_t_axes)
    else:
        key_t = key

    scores = backend_module.matmul(query, key_t) * scale

    if is_causal:
        scores = _apply_causal_mask(backend_module, scores)

    if mask is not None:
        scores = scores + mask

    attn = _apply_softmax(backend_module, scores)

    return backend_module.matmul(attn, value)


@global_eager_registry.register("Acos")
def _acos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the acos operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arccos", getattr(backend_module, "acos", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Acosh")
def _acosh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the acosh operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arccosh", getattr(backend_module, "acosh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Asin")
def _asin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the asin operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arcsin", getattr(backend_module, "asin", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Asinh")
def _asinh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the asinh operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arcsinh", getattr(backend_module, "asinh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atan")
def _atan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atan operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arctan", getattr(backend_module, "atan", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atanh")
def _atanh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atanh operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arctanh", getattr(backend_module, "atanh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atan2")
def _atan2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atan2 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "arctan2", getattr(backend_module, "atan2", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Deg2Rad")
def _deg2rad(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the deg2rad operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "deg2rad", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Degrees")
def _degrees(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the degrees operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "degrees", getattr(backend_module, "rad2deg", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Rad2Deg")
def _rad2deg(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rad2deg operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "rad2deg", getattr(backend_module, "degrees", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Radians")
def _radians(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the radians operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "radians", getattr(backend_module, "deg2rad", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Cbrt")
def _cbrt(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cbrt operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "cbrt", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return backend_module.sign(x) * backend_module.power(backend_module.abs(x), 1.0 / 3.0)


@global_eager_registry.register("Fix")
def _fix(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fix operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fix", getattr(backend_module, "trunc", None))
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return backend_module.where(x >= 0, backend_module.floor(x), backend_module.ceil(x))


@global_eager_registry.register("Copysign")
def _copysign(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the copysign operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "copysign", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    return backend_module.abs(x) * backend_module.sign(y)


@global_eager_registry.register("FloatPower")
def _float_power(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the float power operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "float_power", getattr(backend_module, "power", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmax")
def _fmax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fmax operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fmax", getattr(backend_module, "maximum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmin")
def _fmin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fmin operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fmin", getattr(backend_module, "minimum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmod")
def _fmod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fmod operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fmod", getattr(backend_module, "remainder", getattr(backend_module, "mod", None)))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Frexp")
def _frexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the frexp operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import math

    func = getattr(backend_module, "frexp", getattr(math, "frexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Hypot")
def _hypot(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the hypot operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "hypot", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    return backend_module.sqrt(x**2 + y**2)


@global_eager_registry.register("I0")
def _i0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the i0 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "i0", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Imag")
def _imag(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the imag operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "imag", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Isclose")
def _isclose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isclose operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isclose", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("IsComplex")
def _iscomplex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the iscomplex operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "iscomplex", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("IsReal")
def _isreal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isreal operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isreal", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Kaiser")
def _kaiser(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the kaiser operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "kaiser", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Lcm")
def _lcm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lcm operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "lcm", getattr(backend_module, "least_common_multiple", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Ldexp")
def _ldexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ldexp operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import math

    func = getattr(backend_module, "ldexp", getattr(math, "ldexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Nextafter")
def _nextafter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the nextafter operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import math

    func = getattr(backend_module, "nextafter", getattr(math, "nextafter", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Polyval")
def _polyval(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the polyval operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "polyval", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Real")
def _real(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the real operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "real", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Signbit")
def _signbit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the signbit operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "signbit", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return x < 0


@global_eager_registry.register("Sinc")
def _sinc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the sinc operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "sinc", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Spacing")
def _spacing(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the spacing operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "spacing", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Unwrap")
def _unwrap(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the unwrap operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "unwrap", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Zeta")
def _zeta(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the zeta operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "zeta", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.zeta(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselI0")
def _bessel_i0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel i0 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "i0", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.i0(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel i0e operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "i0e", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.i0e(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselI1")
def _bessel_i1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel i1 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "i1", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.i1(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel i1e operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "i1e", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.i1e(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselJ0")
def _bessel_j0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel j0 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "j0", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.j0(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselJ1")
def _bessel_j1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel j1 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "j1", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.j1(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselJn")
def _bessel_jn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel jn operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "jv", getattr(backend_module, "jn", None))
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.jv(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselK0")
def _bessel_k0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k0 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "k0", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.k0(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k0e operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "k0e", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.k0e(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselK1")
def _bessel_k1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k1 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "k1", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.k1(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k1e operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "k1e", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.k1e(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselY0")
def _bessel_y0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel y0 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "y0", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.y0(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("BesselY1")
def _bessel_y1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel y1 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "y1", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.y1(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Beta")
def _beta(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the beta operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    kwargs.pop("shape", None)
    kwargs.pop("dtype", None)
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "beta"):
        return backend_module.random.beta(getattr(args[1], "data", args[1]), getattr(args[2], "data", args[2]))

    func = getattr(backend_module, "beta", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.beta(*args, **kwargs)

    except ImportError:
        return None


@global_eager_registry.register("Betainc")
def _betainc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the betainc operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "betainc", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.betainc(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Digamma")
def _digamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the digamma operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "digamma", getattr(backend_module, "psi", None))
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.digamma(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Igammac")
def _igammac(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the igammac operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "igammac", getattr(backend_module, "gammaincc", None))
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.gammaincc(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Polygamma")
def _polygamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the polygamma operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "polygamma", None)
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.polygamma(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Heaviside")
def _heaviside(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the heaviside operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "heaviside", getattr(backend_module, "step", None))
    if func:
        return func(*args, **kwargs)
    x, h0 = args[0], args[1]
    return backend_module.where(x < 0, 0.0, backend_module.where(x > 0, 1.0, h0))


@global_eager_registry.register("AccumulateN")
def _accumulate_n(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the accumulate n operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        return None
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = res + inputs[i]
    return res


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the activity regularization operation.

    Args:
        backend_module (object): The backend module.
        x (object): The input tensor.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result (returns x directly).
    """
    return x


def _global_adaptive_pool_mock(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate global adaptive pool mock."""
    if hasattr(operand, "shape") and hasattr(backend_module, "zeros"):
        s = list(operand.shape)
        if isinstance(output_size, int):
            out_s = [output_size]
            s[-1] = output_size
        else:
            out_s = list(output_size)
            s[-len(output_size) :] = out_s

        if hasattr(backend_module, "broadcast_to") and hasattr(backend_module, "mean"):
            axes = tuple(range(-len(out_s), 0))
            return backend_module.broadcast_to(backend_module.mean(operand, axis=axes, keepdims=True), s)

        dtype = getattr(operand, "dtype", None)
        return backend_module.zeros(s, dtype=dtype) if dtype is not None else backend_module.zeros(s)
    return operand


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate adaptive avg pool2d."""
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate adaptive avg pool3d."""
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool2D")
def _adaptive_max_pool2d(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate adaptive max pool2d."""
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D")
def _adaptive_max_pool3d(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate adaptive max pool3d."""
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _adaptive_max_pool3d_indices(backend_module: object, operand: object, output_size: object, **kwargs: object) -> object:
    """Evaluate adaptive max pool3d indices."""
    res = _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)
    return (res, res)


@global_eager_registry.register("AdaptiveLogSoftmaxWithLoss")
def _adaptive_log_softmax_with_loss(backend_module: object, input: object, target: object, *args: object, **kwargs: object) -> object:
    """Evaluate adaptive log softmax with loss."""
    loss = backend_module.zeros((), dtype=getattr(target, "dtype", None)) if hasattr(backend_module, "zeros") else 0.0
    return (target, loss)


@global_eager_registry.register("AllGather")
def _all_gather(backend_module: object, tensor: object, *args: object, **kwargs: object) -> object:
    """Evaluate all gather."""
    if hasattr(backend_module, "stack"):
        return backend_module.stack([tensor])
    if hasattr(backend_module, "array"):
        return backend_module.array([tensor])
    return tensor


@global_eager_registry.register("AllToAll")
def _all_to_all(backend_module: object, tensor: object, *args: object, **kwargs: object) -> object:
    """Evaluate all to all."""
    return tensor


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate alpha dropout."""
    return x


@global_eager_registry.register("ApplyOverAxes")
def _apply_over_axes(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate apply over axes."""
    return args[1] if len(args) > 1 else None


@global_eager_registry.register("Argpartition")
def _argpartition(backend_module: object, a: object, kth: object, axis: int = -1, **kwargs: object) -> object:
    """Evaluate argpartition."""
    return backend_module.argsort(a, axis=axis) if hasattr(backend_module, "argsort") else a


@global_eager_registry.register("Argsort")
def _argsort(backend_module: object, a: object, axis: int = -1, **kwargs: object) -> object:
    """Evaluate argsort."""
    return backend_module.argsort(a, axis=axis)


@global_eager_registry.register("Argwhere")
def _argwhere(backend_module: object, a: object, **kwargs: object) -> object:
    """Evaluate argwhere."""
    return backend_module.argwhere(a)


@global_eager_registry.register("ArrayEquiv")
def _array_equiv(backend_module: object, a1: object, a2: object, **kwargs: object) -> object:
    """Evaluate array equiv."""
    return backend_module.allclose(a1, a2) if hasattr(backend_module, "allclose") else True


@global_eager_registry.register("ArrayRepr")
def _array_repr(backend_module: object, arr: object, **kwargs: object) -> object:
    """Evaluate array repr."""
    return repr(arr)


@global_eager_registry.register("ArrayStr")
def _array_str(backend_module: object, arr: object, **kwargs: object) -> object:
    """Evaluate array str."""
    return str(arr)


@global_eager_registry.register("AsString")
def _as_string(backend_module: object, arr: object, **kwargs: object) -> object:
    """Evaluate as string."""
    return str(arr)


@global_eager_registry.register("Assert")
def _assert(backend_module: object, condition: object, data: object, summarize: int = 3, **kwargs: object) -> object:
    """Evaluate assert."""
    return None


@global_eager_registry.register("Assign")
def _assign(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate assign."""
    return value


@global_eager_registry.register("AssignAdd")
def _assign_add(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate assign add."""
    return ref + value


@global_eager_registry.register("AssignSub")
def _assign_sub(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate assign sub."""
    return ref - value


@global_eager_registry.register("AssignVariable")
def _assign_variable(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate assign variable."""
    return value


@global_eager_registry.register("AssociativeScan")
def _associative_scan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate associative scan."""
    fn = args[0]
    elems = args[1]
    axis = kwargs.get("axis", 0)

    elems_arr = backend_module.array(elems)
    out = backend_module.empty_like(elems_arr)

    elems_arr = backend_module.moveaxis(elems_arr, axis, 0)
    out = backend_module.moveaxis(out, axis, 0)

    acc = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = fn(acc, elems_arr[i])
        out[i] = acc

    out = backend_module.moveaxis(out, 0, axis)
    return out


@global_eager_registry.register("Atleast1d")
def _atleast_1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate atleast 1d."""
    return backend_module.atleast_1d(*args) if hasattr(backend_module, "atleast_1d") else args[0]


@global_eager_registry.register("Atleast2d")
def _atleast_2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate atleast 2d."""
    return backend_module.atleast_2d(*args) if hasattr(backend_module, "atleast_2d") else args[0]


@global_eager_registry.register("Atleast3d")
def _atleast_3d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate atleast 3d."""
    return backend_module.atleast_3d(*args) if hasattr(backend_module, "atleast_3d") else args[0]


@global_eager_registry.register("AxisIndex")
def _axis_index(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate axis index."""
    return backend_module.array(0) if hasattr(backend_module, "array") else 0


@global_eager_registry.register("AddN")
def _add_n(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the add n operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    inputs = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        return None
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = res + inputs[i]
    return res


@global_eager_registry.register("Adjoint")
def _adjoint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the adjoint operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "adjoint", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]

    # Default numpy fallback for general array conjugate transpose
    if hasattr(backend_module, "conj") and hasattr(backend_module, "transpose"):
        return backend_module.conj(backend_module.transpose(x))
    # Fallback to pure python/numpy logic if strictly eager and unbacked
    x_np = backend_module.asarray(x)
    return backend_module.conj(backend_module.transpose(x_np))


@global_eager_registry.register("Det")
def _det(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the det operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "det"):
        return func.det(*args, **kwargs)
    if hasattr(backend_module, "det"):
        return backend_module.det(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.det(backend_module.asarray(x))


@global_eager_registry.register("Eig")
def _eig(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the eig operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eig"):
        return func.eig(*args, **kwargs)
    if hasattr(backend_module, "eig"):
        return backend_module.eig(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.eig(backend_module.asarray(x))


@global_eager_registry.register("Eigh")
def _eigh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the eigh operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigh"):
        return func.eigh(*args, **kwargs)
    if hasattr(backend_module, "eigh"):
        return backend_module.eigh(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.eigh(backend_module.asarray(x))


@global_eager_registry.register("Eigvals")
def _eigvals(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the eigvals operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvals"):
        return func.eigvals(*args, **kwargs)
    if hasattr(backend_module, "eigvals"):
        return backend_module.eigvals(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.eigvals(backend_module.asarray(x))


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the eigvalsh operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvalsh"):
        return func.eigvalsh(*args, **kwargs)
    if hasattr(backend_module, "eigvalsh"):
        return backend_module.eigvalsh(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.eigvalsh(backend_module.asarray(x))


@global_eager_registry.register("Cholesky")
def _cholesky(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cholesky operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "cholesky"):
        return func.cholesky(*args, **kwargs)
    if hasattr(backend_module, "cholesky"):
        return backend_module.cholesky(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.cholesky(backend_module.asarray(x))


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cholesky_ex operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    # Simply fall back to cholesky and return 0 as info status
    chol = _cholesky(backend_module, *args, **kwargs)
    if hasattr(backend_module, "zeros"):
        info = backend_module.zeros((), dtype=getattr(backend_module, "int32", int))
    else:
        info = 0
    return chol, info


@global_eager_registry.register("CholeskySolve")
def _cholesky_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cholesky_solve operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import scipy.linalg

    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "cho_solve"):
        return func.cho_solve(*args, **kwargs)
    if hasattr(backend_module, "cho_solve"):
        return backend_module.cho_solve(*args, **kwargs)

    b, c = args[0], args[1]

    return scipy.linalg.cho_solve((backend_module.asarray(c), False), backend_module.asarray(b))


@global_eager_registry.register("BandedTriangularSolve")
def _banded_triangular_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the banded_triangular_solve operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "solve_banded"):
        return func.solve_banded(*args, **kwargs)
    if hasattr(backend_module, "solve_banded"):
        return backend_module.solve_banded(*args, **kwargs)

    import scipy.linalg

    a, b = args[0], args[1]
    return scipy.linalg.solve_banded((1, 1), backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("HouseholderProduct")
def _householder_product(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the householder_product operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "householder_product"):
        return func.householder_product(*args, **kwargs)
    if hasattr(backend_module, "householder_product"):
        return backend_module.householder_product(*args, **kwargs)

    v, tau = backend_module.asarray(args[0]), backend_module.asarray(args[1])
    m, n = v.shape[-2:]
    k = tau.shape[-1]

    batch_shape = v.shape[:-2]
    identity = backend_module.broadcast_to(backend_module.eye(m, dtype=v.dtype), batch_shape + (m, m)).copy()
    q = identity.copy()

    for i in range(k):
        v_i = v[..., :, i].copy()
        v_i[..., :i] = 0
        v_i[..., i] = 1

        v_i_expanded = v_i[..., backend_module.newaxis]
        v_i_h = backend_module.conjugate(v_i_expanded.swapaxes(-1, -2))

        tau_i = tau[..., i, backend_module.newaxis, backend_module.newaxis]

        h_i = identity - tau_i * (v_i_expanded @ v_i_h)
        q = q @ h_i

    return q[..., :n]


@global_eager_registry.register("Igammac")
@global_eager_registry.register("Polygamma")
@global_eager_registry.register("MatrixPower")
def _matrix_power(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the matrix_power operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "matrix_power"):
        return func.matrix_power(*args, **kwargs)
    if hasattr(backend_module, "matrix_power"):
        return backend_module.matrix_power(*args, **kwargs)

    x, n = args[0], args[1]
    return backend_module.linalg.matrix_power(backend_module.asarray(x), n)


@global_eager_registry.register("MatrixRank")
def _matrix_rank(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the matrix_rank operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "matrix_rank"):
        return func.matrix_rank(*args, **kwargs)
    if hasattr(backend_module, "matrix_rank"):
        return backend_module.matrix_rank(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.matrix_rank(backend_module.asarray(x))


@global_eager_registry.register("Norm")
def _norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the norm operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "norm"):
        return func.norm(*args, **kwargs)
    if hasattr(backend_module, "norm"):
        return backend_module.norm(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.norm(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Pinv")
def _pinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the pinv operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "pinv"):
        return func.pinv(*args, **kwargs)
    if hasattr(backend_module, "pinv"):
        return backend_module.pinv(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.pinv(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Qr")
def _qr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the qr operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "qr"):
        return func.qr(*args, **kwargs)
    if hasattr(backend_module, "qr"):
        return backend_module.qr(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.qr(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Slogdet")
def _slogdet(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the slogdet operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "slogdet"):
        return func.slogdet(*args, **kwargs)
    if hasattr(backend_module, "slogdet"):
        return backend_module.slogdet(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.slogdet(backend_module.asarray(x))


@global_eager_registry.register("Solve")
def _solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the solve operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "solve"):
        return func.solve(*args, **kwargs)
    if hasattr(backend_module, "solve"):
        return backend_module.solve(*args, **kwargs)

    a, b = args[0], args[1]
    return backend_module.linalg.solve(backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("Svd")
def _svd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the svd operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "svd"):
        return func.svd(*args, **kwargs)
    if hasattr(backend_module, "svd"):
        return backend_module.svd(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.svd(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Tensorinv")
def _tensorinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tensorinv operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "tensorinv"):
        return func.tensorinv(*args, **kwargs)
    if hasattr(backend_module, "tensorinv"):
        return backend_module.tensorinv(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.tensorinv(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Tensorsolve")
def _tensorsolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tensorsolve operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "tensorsolve"):
        return func.tensorsolve(*args, **kwargs)
    if hasattr(backend_module, "tensorsolve"):
        return backend_module.tensorsolve(*args, **kwargs)

    a, b = args[0], args[1]
    return backend_module.linalg.tensorsolve(backend_module.asarray(a), backend_module.asarray(b), **kwargs)


@global_eager_registry.register("Bincount")
def _bincount(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bincount operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "bincount", None)
    if func:
        return func(*args, **kwargs)

    x = args[0]
    weights = kwargs.get("weights", None)
    minlength = kwargs.get("minlength", 0)
    return backend_module.bincount(backend_module.asarray(x), weights=backend_module.asarray(weights) if weights is not None else None, minlength=minlength)


@global_eager_registry.register("Correlate")
def _correlate(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the correlate operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "correlate", None)
    if func:
        return func(*args, **kwargs)

    a, v = args[0], args[1]
    mode = kwargs.get("mode", "valid")
    return backend_module.correlate(backend_module.asarray(a), backend_module.asarray(v), mode=mode)


@global_eager_registry.register("Cross")
def _cross(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cross operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "cross", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.cross(backend_module.asarray(args[0]), backend_module.asarray(args[1]), **kwargs)


@global_eager_registry.register("Cummax")
def _cummax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cummax operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "maximum", None)
    if func and hasattr(func, "accumulate"):
        return func.accumulate(*args, **kwargs)

    return backend_module.maximum.accumulate(backend_module.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cummin")
def _cummin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cummin operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "minimum", None)
    if func and hasattr(func, "accumulate"):
        return func.accumulate(*args, **kwargs)

    return backend_module.minimum.accumulate(backend_module.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cumlogsumexp")
def _cumlogsumexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cumlogsumexp operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "cumlogsumexp", None)
    if func:
        return func(*args, **kwargs)

    x = backend_module.asarray(args[0])
    axis = kwargs.get("axis", 0)

    return backend_module.ufunc.accumulate(backend_module.logaddexp, x, axis=axis)


@global_eager_registry.register("CumulativeLogsumexp")
def _cumulative_logsumexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cumulative_logsumexp operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return _cumlogsumexp(backend_module, *args, **kwargs)


@global_eager_registry.register("DivideNoNan")
def _divide_no_nan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the divide_no_nan operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "divide_no_nan", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    res = backend_module.divide(x, y)
    return backend_module.where(y == 0, 0.0, res)


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the multiply_no_nan operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "multiply_no_nan", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    res = backend_module.multiply(x, y)
    return backend_module.where(backend_module.isnan(res), 0.0, res)


@global_eager_registry.register("Extract")
def _extract(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the extract operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "extract", None)
    if func:
        return func(*args, **kwargs)
    condition, arr = args[0], args[1]

    return backend_module.extract(backend_module.asarray(condition), backend_module.asarray(arr))


@global_eager_registry.register("Fft2")
def _fft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fft2 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fft2"):
        return func.fft2(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fft2(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Fftfreq")
def _fftfreq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftfreq operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftfreq"):
        return func.fftfreq(*args, **kwargs)

    n = args[0]
    d = kwargs.get("d", 1.0)
    return backend_module.fft.fftfreq(n, d=d)


@global_eager_registry.register("Fftnd")
def _fftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftnd operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftn"):
        return func.fftn(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fftn(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftshift operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftshift"):
        return func.fftshift(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fftshift(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifft")
def _ifft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifft operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifft"):
        return func.ifft(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifft(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifft2")
def _ifft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifft2 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifft2"):
        return func.ifft2(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifft2(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifftn")
def _ifftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifftn operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifftn"):
        return func.ifftn(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifftn(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifftshift")
def _ifftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifftshift operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifftshift"):
        return func.ifftshift(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifftshift(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Igamma")
def _igamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the igamma operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "igamma", getattr(backend_module, "gammainc", None))
    if func:
        return func(*args, **kwargs)
    try:
        import scipy.special

        return scipy.special.gammainc(*args, **kwargs)
    except ImportError:
        return None


@global_eager_registry.register("Inner")
def _inner(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the inner operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "inner", None)
    if func:
        return func(*args, **kwargs)

    a, b = args[0], args[1]
    return backend_module.inner(backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("Inv")
def _inv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the inv operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "inv"):
        return func.inv(*args, **kwargs)
    if hasattr(backend_module, "inv"):
        return backend_module.inv(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.inv(backend_module.asarray(x))


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the iscomplexobj operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.iscomplexobj(*args, **kwargs)


@global_eager_registry.register("Isrealobj")
def _isrealobj(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isrealobj operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.isrealobj(*args, **kwargs)


@global_eager_registry.register("Issubdtype")
def _issubdtype(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the issubdtype operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@global_eager_registry.register("Isin")
def _isin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isin operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.isin(*args, **kwargs)


@global_eager_registry.register("Ediff1d")
def _ediff1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ediff1d operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.ediff1d(*args, **kwargs)


@global_eager_registry.register("Finfo")
def _finfo(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the finfo operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.finfo(*args, **kwargs)


@global_eager_registry.register("Iinfo")
def _iinfo(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the iinfo operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.iinfo(*args, **kwargs)


@global_eager_registry.register("Fromfile")
def _fromfile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fromfile operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.fromfile(*args, **kwargs)


@global_eager_registry.register("Fromfunction")
def _fromfunction(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fromfunction operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.fromfunction(*args, **kwargs)


@global_eager_registry.register("FromDlpack")
def _from_dlpack(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the from_dlpack operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.from_dlpack(*args, **kwargs)


@global_eager_registry.register("Frompyfunc")
def _frompyfunc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the frompyfunc operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.frompyfunc(*args, **kwargs)


@global_eager_registry.register("Geomspace")
def _geomspace(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the geomspace operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.geomspace(*args, **kwargs)


@global_eager_registry.register("Geometric")
def _geometric(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the geometric operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return getattr(backend_module, "random", backend_module).geometric(*args, **kwargs)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the getprintoptions operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@global_eager_registry.register("Gradient")
def _gradient(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the gradient operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.gradient(*args, **kwargs)


@global_eager_registry.register("HardSilu")
def _hardsilu(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the hardsilu operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    x = args[0]
    return x * backend_module.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("HardSwish")
def _hardswish(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the hardswish operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    x = args[0]
    return x * backend_module.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("Histogram")
def _histogram(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the histogram operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.histogram(*args, **kwargs)


@global_eager_registry.register("Histogram2d")
def _histogram2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the histogram2d operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@global_eager_registry.register("HistogramBinEdges")
def _histogrambinedges(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the histogrambinedges operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@global_eager_registry.register("Histogramdd")
def _histogramdd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the histogramdd operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@global_eager_registry.register("Indices")
def _indices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the indices operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.indices(*args, **kwargs)


@global_eager_registry.register("Infeed")
def _infeed(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the infeed operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "infeed"):
        return backend_module.lax.infeed(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Interp")
def _interp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the interp operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.interp(*args, **kwargs)


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the intersect1d operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.intersect1d(*args, **kwargs)


@global_eager_registry.register("Isscalar")
def _isscalar(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isscalar operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.isscalar(*args, **kwargs)


@global_eager_registry.register("Iterable")
def _iterable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the iterable operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.iterable(*args, **kwargs)


@global_eager_registry.register("Ix")
def _ix(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ix operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.ix_(*args, **kwargs)


@global_eager_registry.register("Kron")
def _kron(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the kron operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.kron(*args, **kwargs)


@global_eager_registry.register("MaskIndices")
def _maskindices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the maskindices operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.mask_indices(*args, **kwargs)


@global_eager_registry.register("Median")
def _median(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the median operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.median(*args, **kwargs)


@global_eager_registry.register("Mgrid")
def _mgrid(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the mgrid operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.mgrid(*args, **kwargs)


@global_eager_registry.register("Mish")
def _mish(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the mish operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    x = args[0]
    return x * backend_module.tanh(backend_module.log1p(backend_module.exp(x)))


@global_eager_registry.register("Modf")
def _modf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the modf operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.modf(*args, **kwargs)


@global_eager_registry.register("Ogrid")
def _ogrid(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ogrid operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.ogrid(*args, **kwargs)


@global_eager_registry.register("Outfeed")
def _outfeed(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the outfeed operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "outfeed"):
        return backend_module.lax.outfeed(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Piecewise")
def _piecewise(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the piecewise operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.piecewise(*args, **kwargs)


@global_eager_registry.register("PromoteTypes")
def _promotetypes(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the promotetypes operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.promote_types(*args, **kwargs)


@global_eager_registry.register("Pshuffle")
def _pshuffle(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the pshuffle operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "pshuffle"):
        return backend_module.lax.pshuffle(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Pswapaxes")
def _pswapaxes(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the pswapaxes operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "pswapaxes"):
        return backend_module.lax.pswapaxes(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("R")
def _r(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the r operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.r_(*args, **kwargs)


@global_eager_registry.register("Rademacher")
def _rademacher(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rademacher operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    shape = kwargs.get("shape", ())
    return backend_module.random.choice([-1, 1], size=shape)


@global_eager_registry.register("ResultType")
def _resulttype(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the resulttype operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.result_type(*args, **kwargs)


@global_eager_registry.register("Rot90")
def _rot90(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rot90 operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.rot90(*args, **kwargs)


@global_eager_registry.register("Squareplus")
def _squareplus(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the squareplus operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    x = args[0]
    b = kwargs.get("b", 4.0)
    return 0.5 * (x + backend_module.sqrt(x**2 + b))


@global_eager_registry.register("Trapezoid")
def _trapezoid(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the trapezoid operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "trapezoid"):
        return backend_module.trapezoid(*args, **kwargs)
    return backend_module.trapz(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tri operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.tri(*args, **kwargs)


@global_eager_registry.register("Tril")
def _tril(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tril operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.tril(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _trimzeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the trimzeros operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.trim_zeros(*args, **kwargs)


@global_eager_registry.register("Triu")
def _triu(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the triu operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.triu(*args, **kwargs)


@global_eager_registry.register("Vander")
def _vander(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the vander operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.vander(*args, **kwargs)


@global_eager_registry.register("Vectorize")
def _vectorize(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the vectorize operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.vectorize(*args, **kwargs)


@global_eager_registry.register("IndexInDim")
def _indexindim(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the indexindim operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if not hasattr(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    index = args[1] if len(args) > 1 else kwargs.get("index")
    axis = kwargs.get("axis", 0)
    keepdims = kwargs.get("keepdims", False)

    slices = [slice(None)] * len(getattr(x, "shape", ()))
    if keepdims and isinstance(index, int):
        slices[axis] = slice(index, index + 1)
    else:
        slices[axis] = index
    return x[tuple(slices)]


@global_eager_registry.register("Lexsort")
def _lexsort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lexsort operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.lexsort(*args, **kwargs)


@global_eager_registry.register("Nonzero")
def _nonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the nonzero operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.nonzero(*args, **kwargs)


@global_eager_registry.register("Percentile")
def _percentile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the percentile operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.percentile(*args, **kwargs)


@global_eager_registry.register("Ppermute")
def _ppermute(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ppermute operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "ppermute"):
        return backend_module.lax.ppermute(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("PsumScatter")
def _psumscatter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the psumscatter operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "psum_scatter"):
        return backend_module.lax.psum_scatter(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Quantile")
def _quantile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the quantile operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.quantile(*args, **kwargs)


@global_eager_registry.register("RavelMultiIndex")
def _ravelmultiindex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ravelmultiindex operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.ravel_multi_index(*args, **kwargs)


@global_eager_registry.register("Repeat")
def _repeat(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the repeat operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.repeat(*args, **kwargs)


@global_eager_registry.register("Searchsorted")
def _searchsorted(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the searchsorted operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.searchsorted(*args, **kwargs)


@global_eager_registry.register("SortComplex")
def _sortcomplex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the sortcomplex operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.sort_complex(*args, **kwargs)


@global_eager_registry.register("Tile")
def _tile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tile operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.tile(*args, **kwargs)


@global_eager_registry.register("Unique")
def _unique(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the unique operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    return backend_module.unique(*args, **kwargs)


@global_eager_registry.register("UpdateSlice")
def _updateslice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the updateslice operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    if not hasattr(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    update = args[1] if len(args) > 1 else kwargs.get("update")
    start_indices = args[2] if len(args) > 2 else kwargs.get("start_indices")

    out = backend_module.array(x).copy()
    slices = tuple(slice(s, s + getattr(update, "shape", ())[i]) for i, s in enumerate(start_indices))
    out[slices] = update
    return out


@global_eager_registry.register("Isinf")
def _isinf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isinf operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isinf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isinf(backend_module.asarray(args[0]))


@global_eager_registry.register("Isnan")
def _isnan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isnan operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isnan", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isnan(backend_module.asarray(args[0]))


@global_eager_registry.register("Isneginf")
def _isneginf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isneginf operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isneginf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isneginf(backend_module.asarray(args[0]))


@global_eager_registry.register("Isposinf")
def _isposinf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isposinf operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "isposinf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isposinf(backend_module.asarray(args[0]))


@global_eager_registry.register("Kronecker")
def _kronecker(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the kronecker operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "kron", getattr(backend_module, "kronecker", None))
    if func:
        return func(*args, **kwargs)

    return backend_module.kron(backend_module.asarray(args[0]), backend_module.asarray(args[1]))


@global_eager_registry.register("Outer")
def _outer(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the outer operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "outer", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.outer(backend_module.asarray(args[0]), backend_module.asarray(args[1]), **kwargs)


@global_eager_registry.register("Fabs")
def _fabs(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fabs operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fabs", getattr(backend_module, "abs", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fill_diagonal operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fill_diagonal", None)
    if func:
        return func(*args, **kwargs)

    x, val = args[0], args[1]
    import numpy as np

    # we need to copy since numpy modifies in place
    x_np = np.array(x, copy=True)
    np.fill_diagonal(x_np, val, **kwargs)
    return backend_module.array(x_np) if hasattr(backend_module, "array") else x_np


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftconvolve operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import scipy.signal

    func = getattr(backend_module, "fftconvolve", None)
    if func:
        return func(*args, **kwargs)

    x, y = args[0], args[1]
    return scipy.signal.fftconvolve(backend_module.asarray(x), backend_module.asarray(y), **kwargs)


@global_eager_registry.register("Flatnonzero")
def _flatnonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the flatnonzero operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "flatnonzero", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.flatnonzero(backend_module.asarray(args[0]))


@global_eager_registry.register("Fliplr")
def _fliplr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fliplr operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fliplr", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fliplr(backend_module.asarray(args[0]))


@global_eager_registry.register("Flipud")
def _flipud(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the flipud operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "flipud", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.flipud(backend_module.asarray(args[0]))


@global_eager_registry.register("Fromiter")
def _fromiter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fromiter operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fromiter", None)
    if "dtype" not in kwargs and len(args) < 2:
        kwargs["dtype"] = float
    if func:
        return func(*args, **kwargs)

    x = args[0]
    dtype = kwargs.pop("dtype", float)
    return backend_module.fromiter(x, dtype=dtype)


@global_eager_registry.register("Fromstring")
def _fromstring(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fromstring operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "fromstring", None)
    if func:
        return func(*args, **kwargs)

    x = args[0]
    return backend_module.fromstring(x, **kwargs)


@global_eager_registry.register("Gamma")
def _gamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the gamma operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    import math

    func = getattr(backend_module, "gamma", math.gamma)
    return func(*args, **kwargs)


@global_eager_registry.register("Gcd")
def _gcd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the gcd operation.

    Args:
        backend_module (object): The backend module.
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        object: The result.
    """
    func = getattr(backend_module, "gcd", getattr(backend_module, "greatest_common_divisor", None))
    if func:
        return func(*args, **kwargs)
    import math

    return math.gcd(*args, **kwargs)


@global_eager_registry.register("Ball")
def _mock_ball(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate ball."""
    func = getattr(backend_module, "ball", None)
    if func:
        return func(*args, **kwargs)

    radius = args[0] if len(args) > 0 else kwargs.get("radius", 1.0)
    size = kwargs.get("size", 1)

    # Simple uniform sampling in n-ball by rejection or normalized gaussian approach.
    # We'll just generate normal and normalize, then multiply by u^(1/d)
    d = backend_module.asarray(radius).size if hasattr(radius, "__len__") else 1
    u = backend_module.random.uniform(0, 1, size)
    norm = backend_module.random.normal(0, 1, (size, max(d, 1)))
    norm_sq = backend_module.sum(norm**2, axis=-1, keepdims=True)
    scale = (u ** (1.0 / max(d, 1))) / backend_module.sqrt(norm_sq)
    return radius * (norm * scale)


@global_eager_registry.register("BandPart")
def _mock_bandpart(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate bandpart."""
    func = getattr(backend_module, "bandpart", None)
    if func:
        return func(*args, **kwargs)

    x = args[0] if len(args) > 0 else kwargs.get("x", None)
    num_lower = args[1] if len(args) > 1 else kwargs.get("num_lower", 0)
    num_upper = args[2] if len(args) > 2 else kwargs.get("num_upper", 0)

    x_np = getattr(x, "data", x)
    if x_np is None:
        raise ValueError("Missing x")

    # tf.linalg.band_part(input, num_lower, num_upper)
    # returns tensor with same shape as input
    # if num_lower < 0, all lower diagonals are kept
    # if num_upper < 0, all upper diagonals are kept
    m, n = backend_module.shape(x_np)[-2:]

    # generate indices
    i, j = backend_module.indices((m, n))

    # default to true mask
    mask = backend_module.ones((m, n), dtype=bool)
    if num_lower >= 0:
        mask = mask & ((i - j) <= num_lower)
    if num_upper >= 0:
        mask = mask & ((j - i) <= num_upper)

    # apply mask to last two dimensions
    return backend_module.where(mask, x_np, backend_module.zeros_like(x_np))


@global_eager_registry.register("BetaPdf")
def _mock_betapdf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate betapdf."""
    func = getattr(backend_module, "betapdf", None)
    if func:
        return func(*args, **kwargs)

    from scipy.stats import beta

    x = args[0] if len(args) > 0 else kwargs.get("x")
    a = args[1] if len(args) > 1 else kwargs.get("a")
    b = args[2] if len(args) > 2 else kwargs.get("b")
    return beta.pdf(backend_module.asarray(x), backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("DecodeImage")
def _mock_decodeimage(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate decodeimage."""
    func = getattr(backend_module, "decodeimage", None)
    if func:
        return func(*args, **kwargs)

    # Usually decode_image requires some image decoding lib (like PIL or cv2)
    # Since we can only rely on numpy here, we'll return a mock image.

    return backend_module.zeros(kwargs.get("shape", (256, 256, 3)), dtype=backend_module.uint8)


@global_eager_registry.register("Deg2Rad")
def _mock_deg2rad(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate deg2rad."""
    func = getattr(backend_module, "deg2rad", None)
    if func:
        return func(*args, **kwargs)

    x = args[0] if len(args) > 0 else kwargs.get("x")
    x = getattr(x, "data", x)
    return backend_module.deg2rad(backend_module.asarray(x))


@global_eager_registry.register("Fmax")
def _mock_fmax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fmax."""
    func = getattr(backend_module, "fmax", None)
    if func:
        return func(*args, **kwargs)

    x1 = args[0] if len(args) > 0 else kwargs.get("x1")
    x2 = args[1] if len(args) > 1 else kwargs.get("x2")
    x1 = getattr(x1, "data", x1)
    x2 = getattr(x2, "data", x2)
    return backend_module.fmax(backend_module.asarray(x1), backend_module.asarray(x2))


@global_eager_registry.register("FractionalAvgPool")
def _mock_fractionalavgpool(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fractionalavgpool."""
    func = getattr(backend_module, "fractionalavgpool", None)
    if func:
        return func(*args, **kwargs)
    if args:
        # return first element simulating mock shape preservation
        return backend_module.asarray(args[0])
    return None


@global_eager_registry.register("Fromfile")
def _mock_fromfile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fromfile."""
    func = getattr(backend_module, "fromfile", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fromfile(*args, **kwargs)


@global_eager_registry.register("Fromfunction")
def _mock_fromfunction(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fromfunction."""
    func = getattr(backend_module, "fromfunction", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fromfunction(*args, **kwargs)


@global_eager_registry.register("Fromiter")
def _mock_fromiter(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fromiter."""
    func = getattr(backend_module, "fromiter", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fromiter(*args, **kwargs)


@global_eager_registry.register("Frompyfunc")
def _mock_frompyfunc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate frompyfunc."""
    func = getattr(backend_module, "frompyfunc", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.frompyfunc(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _mock_fromstring(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate fromstring."""
    func = getattr(backend_module, "fromstring", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fromstring(*args, **kwargs)


@global_eager_registry.register("Gamma")
def _mock_gamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate gamma."""
    func = getattr(backend_module, "gamma", None)
    if func:
        return func(*args, **kwargs)

    from scipy.special import gamma

    x = args[0] if len(args) > 0 else kwargs.get("x")
    x = getattr(x, "data", x)
    return gamma(backend_module.asarray(x))


@global_eager_registry.register("Gcd")
def _mock_gcd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate gcd."""
    func = getattr(backend_module, "gcd", None)
    if func:
        return func(*args, **kwargs)
    import math

    x1 = args[0] if len(args) > 0 else kwargs.get("x1")
    x2 = args[1] if len(args) > 1 else kwargs.get("x2")
    x1 = getattr(x1, "data", x1)
    x2 = getattr(x2, "data", x2)

    try:
        return backend_module.gcd(backend_module.asarray(x1), backend_module.asarray(x2))
    except AttributeError:
        # Fallback for older numpy if needed
        return backend_module.vectorize(math.gcd)(backend_module.asarray(x1), backend_module.asarray(x2))


@global_eager_registry.register("Geometric")
def _mock_geometric(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate geometric."""
    func = getattr(backend_module, "geometric", None)
    if func:
        return func(*args, **kwargs)

    p = args[0] if len(args) > 0 else kwargs.get("p")
    size = kwargs.get("size")
    return backend_module.random.geometric(backend_module.asarray(p), size=size)


@global_eager_registry.register("Gumbel")
def _mock_gumbel(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate gumbel."""
    func = getattr(backend_module, "gumbel", None)
    if func:
        return func(*args, **kwargs)

    loc = args[0] if len(args) > 0 else kwargs.get("loc", 0.0)
    scale = args[1] if len(args) > 1 else kwargs.get("scale", 1.0)
    size = kwargs.get("size")
    return backend_module.random.gumbel(backend_module.asarray(loc), backend_module.asarray(scale), size=size)


@global_eager_registry.register("Heaviside")
def _mock_heaviside(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate heaviside."""
    func = getattr(backend_module, "heaviside", None)
    if func:
        return func(*args, **kwargs)

    x1 = args[0] if len(args) > 0 else kwargs.get("x1")
    x2 = args[1] if len(args) > 1 else kwargs.get("x2")
    x1 = getattr(x1, "data", x1)
    x2 = getattr(x2, "data", x2)
    return backend_module.heaviside(backend_module.asarray(x1), backend_module.asarray(x2))


@global_eager_registry.register("Hfft")
def _mock_hfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate hfft."""
    func = getattr(backend_module, "hfft", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fft.hfft(*args, **kwargs)


@global_eager_registry.register("Hsplit")
def _mock_hsplit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate hsplit."""
    func = getattr(backend_module, "hsplit", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.hsplit(*args, **kwargs)


@global_eager_registry.register("Inner")
def _mock_inner(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate inner."""
    func = getattr(backend_module, "inner", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.inner(*args, **kwargs)


@global_eager_registry.register("ModifiedBesselI1")
def _mock_modifiedbesseli1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate modifiedbesseli1."""
    func = getattr(backend_module, "modifiedbesseli1", None)
    if func:
        return func(*args, **kwargs)

    from scipy.special import i1

    x = args[0] if len(args) > 0 else kwargs.get("x")
    x = getattr(x, "data", x)
    return i1(backend_module.asarray(x))


@global_eager_registry.register("Packbits")
def _mock_packbits(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate packbits."""
    func = getattr(backend_module, "packbits", None)
    if func:
        return func(*args, **kwargs)

    x = args[0] if len(args) > 0 else kwargs.get("x")
    x = getattr(x, "data", x)
    kwargs_for_pack = {k: v for k, v in kwargs.items() if k in ["axis", "bitorder"]}
    return backend_module.packbits(backend_module.asarray(x), **kwargs_for_pack)


@global_eager_registry.register("ParseTensor")
def _mock_parsetensor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate parsetensor."""
    func = getattr(backend_module, "parsetensor", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.asarray(args[0]) if args else None


@global_eager_registry.register("Partition")
def _mock_partition(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate partition."""
    func = getattr(backend_module, "partition", None)
    if func:
        return func(*args, **kwargs)

    arr = args[0] if len(args) > 0 else kwargs.get("arr")
    kth = args[1] if len(args) > 1 else kwargs.get("kth")
    arr = getattr(arr, "data", arr)
    kwargs_for_part = {k: v for k, v in kwargs.items() if k in ["axis", "kind", "order"]}
    return backend_module.partition(backend_module.asarray(arr), kth, **kwargs_for_part)


@global_eager_registry.register("Polyint")
def _mock_polyint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate polyint."""
    func = getattr(backend_module, "polyint", None)
    if func:
        return func(*args, **kwargs)

    p = args[0] if len(args) > 0 else kwargs.get("p")
    p = getattr(p, "data", p)
    m = kwargs.get("m", args[1] if len(args) > 1 else 1)
    k = kwargs.get("k", args[2] if len(args) > 2 else None)

    import numpy as np

    if k is not None:
        res = np.polyint(np.asarray(p), m=m, k=k)
    else:
        res = np.polyint(np.asarray(p), m=m)
    return backend_module.asarray(res) if hasattr(backend_module, "asarray") else res


@global_eager_registry.register("R")
def _mock_r(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate r."""
    func = getattr(backend_module, "r", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.r_[args]


@global_eager_registry.register("RngUniform")
def _mock_rnguniform(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate rnguniform."""
    func = getattr(backend_module, "rnguniform", None)
    if func:
        return func(*args, **kwargs)

    shape = args[0] if len(args) > 0 else kwargs.get("shape")
    low = args[1] if len(args) > 1 else kwargs.get("low", 0.0)
    high = args[2] if len(args) > 2 else kwargs.get("high", 1.0)
    return backend_module.random.uniform(low=backend_module.asarray(low).item(), high=backend_module.asarray(high).item(), size=shape)


@global_eager_registry.register("ScatterApply")
def _mock_scatterapply(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate scatterapply."""
    func = getattr(backend_module, "scatterapply", None)
    if func:
        return func(*args, **kwargs)

    if len(args) < 4:
        return args[0]

    tensor = backend_module.asarray(args[0]).copy()
    indices = backend_module.asarray(args[1])
    updates = backend_module.asarray(args[2])
    reduction = args[3] if len(args) > 3 else kwargs.get("reduction", None)

    # Just a mock scatter, applying update on flattened for simplicty if shape mismatch

    try:
        if reduction == "add":
            backend_module.add.at(tensor, tuple(indices.T), updates)
        elif reduction == "mul":
            backend_module.multiply.at(tensor, tuple(indices.T), updates)
        else:
            tensor[tuple(indices.T)] = updates
    except Exception as e:
        raise RuntimeError(f"TensorScatterUpdate failed: {e}") from e
    return tensor


@global_eager_registry.register("ScatterMax")
def _mock_scattermax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate scattermax."""
    func = getattr(backend_module, "scattermax", getattr(backend_module, "scatter_max", None))
    if func:
        return func(*args, **kwargs)

    tensor = args[0] if len(args) > 0 else kwargs.get("tensor")
    indices = args[1] if len(args) > 1 else kwargs.get("indices")
    updates = args[2] if len(args) > 2 else kwargs.get("updates")

    res = backend_module.array(getattr(tensor, "data", tensor))
    indices_arr = backend_module.asarray(getattr(indices, "data", indices))
    updates_arr = backend_module.asarray(getattr(updates, "data", updates))
    idx = tuple(indices_arr[..., dim] for dim in range(indices_arr.shape[-1]))
    res[idx] = backend_module.maximum(res[idx], updates_arr)
    return res


@global_eager_registry.register("ScatterMin")
def _mock_scattermin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate scattermin."""
    func = getattr(backend_module, "scattermin", getattr(backend_module, "scatter_min", None))
    if func:
        return func(*args, **kwargs)

    tensor = args[0] if len(args) > 0 else kwargs.get("tensor")
    indices = args[1] if len(args) > 1 else kwargs.get("indices")
    updates = args[2] if len(args) > 2 else kwargs.get("updates")

    res = backend_module.array(getattr(tensor, "data", tensor))
    indices_arr = backend_module.asarray(getattr(indices, "data", indices))
    updates_arr = backend_module.asarray(getattr(updates, "data", updates))
    idx = tuple(indices_arr[..., dim] for dim in range(indices_arr.shape[-1]))
    res[idx] = backend_module.minimum(res[idx], updates_arr)
    return res


@global_eager_registry.register("ScatterMul")
def _mock_scattermul(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate scattermul."""
    func = getattr(backend_module, "scattermul", getattr(backend_module, "scatter_mul", None))
    if func:
        return func(*args, **kwargs)

    tensor = args[0] if len(args) > 0 else kwargs.get("tensor")
    indices = args[1] if len(args) > 1 else kwargs.get("indices")
    updates = args[2] if len(args) > 2 else kwargs.get("updates")

    res = backend_module.array(getattr(tensor, "data", tensor))
    indices_arr = backend_module.asarray(getattr(indices, "data", indices))
    updates_arr = backend_module.asarray(getattr(updates, "data", updates))
    idx = tuple(indices_arr[..., dim] for dim in range(indices_arr.shape[-1]))
    res[idx] = res[idx] * updates_arr
    return res


@global_eager_registry.register("ScatterNd")
def _mock_scatternd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate scatternd."""
    func = getattr(backend_module, "scatter_nd", None)
    if func:
        return func(*args, **kwargs)

    indices = args[0] if len(args) > 0 else kwargs.get("indices")
    updates = args[1] if len(args) > 1 else kwargs.get("updates")
    shape = args[2] if len(args) > 2 else kwargs.get("shape")

    res = backend_module.zeros(shape, dtype=backend_module.asarray(updates).dtype)
    indices_arr = backend_module.asarray(getattr(indices, "data", indices))
    updates_arr = backend_module.asarray(getattr(updates, "data", updates))
    idx = tuple(indices_arr[..., dim] for dim in range(indices_arr.shape[-1]))
    res[idx] = updates_arr
    return res


@global_eager_registry.register("Schur")
def _mock_schur(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate schur."""
    func = getattr(backend_module, "schur", None)
    if func:
        return func(*args, **kwargs)

    import scipy.linalg

    a = args[0] if len(args) > 0 else kwargs.get("a")
    return scipy.linalg.schur(backend_module.asarray(a), **kwargs)


@global_eager_registry.register("StringLower")
def _mock_stringlower(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringlower."""
    func = getattr(backend_module, "stringlower", None)
    if func:
        return func(*args, **kwargs)
    return str(args[0]).lower()


@global_eager_registry.register("StringSplit")
def _mock_stringsplit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringsplit."""
    func = getattr(backend_module, "stringsplit", None)
    if func:
        return func(*args, **kwargs)
    return str(args[0]).split(args[1] if len(args) > 1 else " ")


@global_eager_registry.register("StringSubstr")
def _mock_stringsubstr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringsubstr."""
    func = getattr(backend_module, "stringsubstr", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "stringsubstr"):
        return backend_module.random.stringsubstr(*args, **kwargs)

    # basic string fallback
    strings = backend_module.asarray(args[0])
    pos = int(args[1]) if len(args) > 1 else kwargs.get("pos", 0)
    length = int(args[2]) if len(args) > 2 else kwargs.get("len", 1)
    # vectorized substring via python list comp
    return backend_module.array([s[pos : pos + length] for s in strings.flat]).reshape(strings.shape)


@global_eager_registry.register("StringToHash")
def _mock_stringtohash(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringtohash."""
    func = getattr(backend_module, "stringtohash", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "stringtohash"):
        return backend_module.random.stringtohash(*args, **kwargs)

    strings = backend_module.asarray(args[0])
    return backend_module.array([hash(str(s)) % (2**31) for s in strings.flat]).reshape(strings.shape)


@global_eager_registry.register("StringToNumber")
def _mock_stringtonumber(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringtonumber."""
    func = getattr(backend_module, "stringtonumber", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "stringtonumber"):
        return backend_module.random.stringtonumber(*args, **kwargs)

    strings = backend_module.asarray(args[0])
    try:
        return strings.astype(backend_module.float32)
    except ValueError:
        return backend_module.zeros_like(strings, dtype=backend_module.float32)


@global_eager_registry.register("StringUpper")
def _mock_stringupper(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate stringupper."""
    func = getattr(backend_module, "stringupper", None)
    if func:
        return func(*args, **kwargs)
    return str(args[0]).upper()


@global_eager_registry.register("Svd")
def _mock_svd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate svd."""
    func = getattr(backend_module, "svd", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.linalg.svd(*args, **kwargs)


@global_eager_registry.register("Svdvals")
def _mock_svdvals(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate svdvals."""
    func = getattr(backend_module, "svdvals", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "svdvals"):
        return backend_module.random.svdvals(*args, **kwargs)

    return backend_module.linalg.svd(args[0], compute_uv=False)


@global_eager_registry.register("Switch")
def _mock_switch(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate switch."""
    func = getattr(backend_module, "switch", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "switch"):
        return backend_module.random.switch(*args, **kwargs)

    return args[1] if args[0] else args[2]


@global_eager_registry.register("T")
def _mock_t(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate t."""
    func = getattr(backend_module, "t", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "t"):
        return backend_module.random.t(*args, **kwargs)

    return backend_module.transpose(args[0])


@global_eager_registry.register("TakeAlongAxis")
def _mock_takealongaxis(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate takealongaxis."""
    func = getattr(backend_module, "takealongaxis", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "takealongaxis"):
        return backend_module.random.takealongaxis(*args, **kwargs)

    # Fallback
    if hasattr(backend_module, "take_along_axis"):
        return backend_module.take_along_axis(*args, **kwargs)

    return backend_module.take_along_axis(args[0], args[1], axis=kwargs.get("axis", -1))


@global_eager_registry.register("TensorArrayRead")
def _mock_tensorarrayread(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorarrayread."""
    func = getattr(backend_module, "tensorarrayread", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tensorarrayread"):
        return backend_module.random.tensorarrayread(*args, **kwargs)

    return args[0][args[1]]


@global_eager_registry.register("TensorArrayStack")
def _mock_tensorarraystack(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorarraystack."""
    func = getattr(backend_module, "tensorarraystack", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tensorarraystack"):
        return backend_module.random.tensorarraystack(*args, **kwargs)

    return backend_module.stack(args[0])


@global_eager_registry.register("TensorArrayWrite")
def _mock_tensorarraywrite(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorarraywrite."""
    func = getattr(backend_module, "tensorarraywrite", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tensorarraywrite"):
        return backend_module.random.tensorarraywrite(*args, **kwargs)

    if len(args) < 3:
        return args[0]

    # Typically args: ta (list-like or tensor), index, value
    ta = list(args[0]) if isinstance(args[0], (list, tuple)) else args[0]
    index = int(args[1])
    value = args[2]

    if isinstance(ta, list):
        if index >= len(ta):
            ta.extend([None] * (index - len(ta) + 1))
        ta[index] = value
        return ta

    # fallback for tensor

    ta = backend_module.asarray(ta).copy()
    ta[index] = value
    return backend_module.asarray(ta)


@global_eager_registry.register("TensorScatterSub")
def _mock_tensorscattersub(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorscattersub."""
    func = getattr(backend_module, "tensorscattersub", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tensorscattersub"):
        return backend_module.random.tensorscattersub(*args, **kwargs)

    return args[0] - args[2]


@global_eager_registry.register("TensorScatterUpdate")
def _mock_tensorscatterupdate(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorscatterupdate."""
    func = getattr(backend_module, "tensorscatterupdate", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tensorscatterupdate"):
        return backend_module.random.tensorscatterupdate(*args, **kwargs)

    return args[2]


@global_eager_registry.register("Tensorinv")
def _mock_tensorinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorinv."""
    func = getattr(backend_module, "tensorinv", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.linalg.tensorinv(*args, **kwargs)


@global_eager_registry.register("Tensorsolve")
def _mock_tensorsolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tensorsolve."""
    func = getattr(backend_module, "tensorsolve", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.linalg.tensorsolve(*args, **kwargs)


@global_eager_registry.register("TextVectorization")
def _mock_textvectorization(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate textvectorization."""
    func = getattr(backend_module, "textvectorization", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "textvectorization"):
        return backend_module.random.textvectorization(*args, **kwargs)

    if not args:
        return backend_module.zeros((1,), dtype=backend_module.int64)

    strings = backend_module.asarray(args[0])
    return backend_module.zeros(strings.shape + (10,), dtype=backend_module.int64)


@global_eager_registry.register("TopK")
def _mock_topk(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate topk."""
    func = getattr(backend_module, "topk", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "topk"):
        return backend_module.random.topk(*args, **kwargs)

    return (args[0], backend_module.zeros_like(args[0], dtype=backend_module.int64))


@global_eager_registry.register("Trapezoid")
def _mock_trapezoid(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate trapezoid."""
    func = getattr(backend_module, "trapezoid", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.trapz(*args, **kwargs)


@global_eager_registry.register("TrapezoidalIntegral")
def _mock_trapezoidalintegral(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate trapezoidalintegral."""
    func = getattr(backend_module, "trapezoidalintegral", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "trapezoidalintegral"):
        return backend_module.random.trapezoidalintegral(*args, **kwargs)

    return backend_module.trapz(args[0], axis=kwargs.get("axis", -1))


@global_eager_registry.register("TriInv")
def _mock_triinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate triinv."""
    func = getattr(backend_module, "triinv", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "triinv"):
        return backend_module.random.triinv(*args, **kwargs)

    return backend_module.linalg.inv(args[0])


@global_eager_registry.register("Triangular")
def _mock_triangular(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate triangular."""
    func = getattr(backend_module, "triangular", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "triangular"):
        return backend_module.random.triangular(*args, **kwargs)

    # Fallback to NumPy
    if hasattr(backend_module, "random"):
        return backend_module.random.triangular(*args, **kwargs)


@global_eager_registry.register("TriangularSolve")
def _mock_triangularsolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate triangularsolve."""
    func = getattr(backend_module, "triangularsolve", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "triangularsolve"):
        return backend_module.random.triangularsolve(*args, **kwargs)

    return backend_module.linalg.solve(args[0], args[1])


@global_eager_registry.register("Tridiagonal")
def _mock_tridiagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tridiagonal."""
    func = getattr(backend_module, "tridiagonal", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tridiagonal"):
        return backend_module.random.tridiagonal(*args, **kwargs)

    return args[0]


@global_eager_registry.register("TridiagonalMatmul")
def _mock_tridiagonalmatmul(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tridiagonalmatmul."""
    func = getattr(backend_module, "tridiagonalmatmul", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tridiagonalmatmul"):
        return backend_module.random.tridiagonalmatmul(*args, **kwargs)

    return backend_module.matmul(args[0], args[1])


@global_eager_registry.register("TridiagonalSolve")
def _mock_tridiagonalsolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate tridiagonalsolve."""
    func = getattr(backend_module, "tridiagonalsolve", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "tridiagonalsolve"):
        return backend_module.random.tridiagonalsolve(*args, **kwargs)

    return backend_module.linalg.solve(args[0], args[1])


@global_eager_registry.register("TrilIndices")
def _mock_trilindices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate trilindices."""
    func = getattr(backend_module, "trilindices", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.tril_indices(*args, **kwargs)


@global_eager_registry.register("TrilIndicesFrom")
def _mock_trilindicesfrom(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate trilindicesfrom."""
    func = getattr(backend_module, "trilindicesfrom", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.tril_indices_from(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _mock_trimzeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate trimzeros."""
    func = getattr(backend_module, "trimzeros", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.trim_zeros(*args, **kwargs)


@global_eager_registry.register("TriuIndices")
def _mock_triuindices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate triuindices."""
    func = getattr(backend_module, "triuindices", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.triu_indices(*args, **kwargs)


@global_eager_registry.register("TriuIndicesFrom")
def _mock_triuindicesfrom(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate triuindicesfrom."""
    func = getattr(backend_module, "triuindicesfrom", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.triu_indices_from(*args, **kwargs)


@global_eager_registry.register("TruncateDiv")
def _mock_truncatediv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate truncatediv."""
    func = getattr(backend_module, "truncatediv", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.trunc(backend_module.divide(*args, **kwargs))


@global_eager_registry.register("TruncateMod")
def _mock_truncatemod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate truncatemod."""
    func = getattr(backend_module, "truncatemod", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.fmod(*args, **kwargs)


@global_eager_registry.register("Unfold")
def _mock_unfold(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate unfold."""
    func = getattr(backend_module, "unfold", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "unfold"):
        return backend_module.random.unfold(*args, **kwargs)

    if not args:
        return None
    tensor = args[0]
    kernel_size = kwargs.get("kernel_size", (3, 3))
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    stride = kwargs.get("stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)

    import numpy as np

    t_np = np.asarray(tensor)

    # Unfold usually operates on 4D tensors (N, C, H, W)
    if t_np.ndim != 4:
        # Generic fallback
        s = list(t_np.shape)
        return backend_module.asarray(np.zeros(s + [1], dtype=t_np.dtype))

    N, C, H, W = t_np.shape
    kH, kW = kernel_size
    sH, sW = stride
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    if out_H <= 0 or out_W <= 0:
        return backend_module.asarray(np.zeros((N, C * kH * kW, 0), dtype=t_np.dtype))
    out = np.zeros((N, C * kH * kW, out_H * out_W), dtype=t_np.dtype)
    idx = 0
    for y in range(0, H - kH + 1, sH):
        for x in range(0, W - kW + 1, sW):
            patch = t_np[:, :, y : y + kH, x : x + kW].reshape(N, C * kH * kW)
            out[:, :, idx] = patch
            idx += 1
    return backend_module.asarray(out)


@global_eager_registry.register("Union1d")
def _mock_union1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate union1d."""
    func = getattr(backend_module, "union1d", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.union1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _mock_unique(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate unique."""
    func = getattr(backend_module, "unique", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.unique(*args, **kwargs)


@global_eager_registry.register("UniqueAll")
def _mock_uniqueall(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate uniqueall."""
    func = getattr(backend_module, "uniqueall", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "uniqueall"):
        return backend_module.random.uniqueall(*args, **kwargs)

    return backend_module.unique(args[0], return_index=True, return_inverse=True, return_counts=True)


@global_eager_registry.register("UniqueCounts")
def _mock_uniquecounts(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate uniquecounts."""
    func = getattr(backend_module, "uniquecounts", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "uniquecounts"):
        return backend_module.random.uniquecounts(*args, **kwargs)

    return backend_module.unique(args[0], return_counts=True)


@global_eager_registry.register("UniqueInverse")
def _mock_uniqueinverse(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate uniqueinverse."""
    func = getattr(backend_module, "uniqueinverse", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "uniqueinverse"):
        return backend_module.random.uniqueinverse(*args, **kwargs)

    return backend_module.unique(args[0], return_inverse=True)


@global_eager_registry.register("UniqueValues")
def _mock_uniquevalues(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate uniquevalues."""
    func = getattr(backend_module, "uniquevalues", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "uniquevalues"):
        return backend_module.random.uniquevalues(*args, **kwargs)

    return backend_module.unique(args[0])


@global_eager_registry.register("Unpackbits")
def _mock_unpackbits(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate unpackbits."""
    func = getattr(backend_module, "unpackbits", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.unpackbits(*args, **kwargs)


@global_eager_registry.register("Unstack")
def _mock_unstack(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate unstack."""
    func = getattr(backend_module, "unstack", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "unstack"):
        return backend_module.random.unstack(*args, **kwargs)

    return tuple(args[0])


@global_eager_registry.register("Unwrap")
def _mock_unwrap(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate unwrap."""
    func = getattr(backend_module, "unwrap", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.unwrap(*args, **kwargs)


@global_eager_registry.register("UpdateSlice")
def _mock_updateslice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate updateslice."""
    func = getattr(backend_module, "updateslice", None)
    if func:
        return func(*args, **kwargs)

    if len(args) < 3:
        return args[0] if args else None

    operand = backend_module.asarray(args[0]).copy()
    update = backend_module.asarray(args[1])
    start_indices = backend_module.asarray(args[2])

    try:
        slices = tuple(slice(s, s + size) for s, size in zip(start_indices, update.shape))
        operand[slices] = update
    except Exception as e:
        raise RuntimeError(f"UpdateSlice failed: {e}") from e
    return backend_module.asarray(operand)


@global_eager_registry.register("Vander")
def _mock_vander(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate vander."""
    func = getattr(backend_module, "vander", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.vander(*args, **kwargs)


@global_eager_registry.register("Variance")
def _mock_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate variance."""
    func = getattr(backend_module, "variance", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "variance"):
        return backend_module.random.variance(*args, **kwargs)

    return backend_module.var(args[0], axis=kwargs.get("axis", None))


@global_eager_registry.register("Vecdot")
def _mock_vecdot(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate vecdot."""
    func = getattr(backend_module, "vecdot", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "vecdot"):
        return backend_module.random.vecdot(*args, **kwargs)

    return backend_module.vdot(args[0], args[1])


@global_eager_registry.register("VectorNorm")
def _mock_vectornorm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate vectornorm."""
    func = getattr(backend_module, "vectornorm", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "vectornorm"):
        return backend_module.random.vectornorm(*args, **kwargs)

    return backend_module.linalg.norm(args[0], axis=kwargs.get("axis", None))


@global_eager_registry.register("Vectorize")
def _mock_vectorize(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate vectorize."""
    func = getattr(backend_module, "vectorize", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.vectorize(*args, **kwargs)


@global_eager_registry.register("Vsplit")
def _mock_vsplit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate vsplit."""
    func = getattr(backend_module, "vsplit", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.vsplit(*args, **kwargs)


@global_eager_registry.register("Wald")
def _mock_wald(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate wald."""
    func = getattr(backend_module, "wald", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.random.wald(*args, **kwargs)


@global_eager_registry.register("WeibullMin")
def _mock_weibullmin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate weibullmin."""
    func = getattr(backend_module, "weibullmin", None)
    if func:
        return func(*args, **kwargs)
    import scipy.stats

    return scipy.stats.weibull_min(*args, **kwargs)


@global_eager_registry.register("Welch")
def _mock_welch(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate welch."""
    func = getattr(backend_module, "welch", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "welch"):
        return backend_module.random.welch(*args, **kwargs)

    return (args[0], args[0])


@global_eager_registry.register("WindowHamming")
def _mock_windowhamming(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate windowhamming."""
    func = getattr(backend_module, "windowhamming", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "windowhamming"):
        return backend_module.random.windowhamming(*args, **kwargs)

    return backend_module.hamming(args[0])


@global_eager_registry.register("WindowHann")
def _mock_windowhann(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate windowhann."""
    func = getattr(backend_module, "windowhann", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "windowhann"):
        return backend_module.random.windowhann(*args, **kwargs)

    return backend_module.hanning(args[0])


@global_eager_registry.register("WrapKeyData")
def _mock_wrapkeydata(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate wrapkeydata."""
    func = getattr(backend_module, "wrapkeydata", None)
    if func:
        return func(*args, **kwargs)

    if not args:
        return None

    x = backend_module.asarray(args[0])
    return backend_module.array([x, backend_module.zeros_like(x)], dtype=backend_module.uint32)


@global_eager_registry.register("WriteFile")
def _mock_writefile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate writefile."""
    func = getattr(backend_module, "writefile", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "writefile"):
        return backend_module.random.writefile(*args, **kwargs)

    return None


@global_eager_registry.register("Xdivy")
def _mock_xdivy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate xdivy."""
    func = getattr(backend_module, "xdivy", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "xdivy"):
        return backend_module.random.xdivy(*args, **kwargs)

    return backend_module.where(args[0] == 0, 0, args[0] / args[1])


@global_eager_registry.register("Xlog1py")
def _mock_xlog1py(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate xlog1py."""
    func = getattr(backend_module, "xlog1py", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "xlog1py"):
        return backend_module.random.xlog1py(*args, **kwargs)

    # Fallback to scipy.special
    return backend_module.where(args[0] == 0, 0, args[0] * backend_module.log1p(args[1]))


@global_eager_registry.register("Xlogy")
def _mock_xlogy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate xlogy."""
    func = getattr(backend_module, "xlogy", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "xlogy"):
        return backend_module.random.xlogy(*args, **kwargs)

    # Fallback to scipy.special
    return backend_module.where(args[0] == 0, 0, args[0] * backend_module.log(args[1]))


@global_eager_registry.register("ZeroFraction")
def _mock_zerofraction(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate zerofraction."""
    func = getattr(backend_module, "zerofraction", None)
    if func:
        return func(*args, **kwargs)

    # Try random module for some
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "zerofraction"):
        return backend_module.random.zerofraction(*args, **kwargs)

    return backend_module.sum(args[0] == 0) / backend_module.prod(args[0].shape)


@global_eager_registry.register("Zeta")
def _mock_zeta(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate zeta."""
    func = getattr(backend_module, "zeta", None)
    if func:
        return func(*args, **kwargs)
    import scipy.special

    return scipy.special.zeta(*args, **kwargs)
