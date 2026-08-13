# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_misc_ext module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Deg2Rad")
def _deg2rad(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _deg2rad operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "deg2rad", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Degrees")
def _degrees(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _degrees operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "degrees", getattr(backend_module, "rad2deg", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Rad2Deg")
def _rad2deg(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _rad2deg operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "rad2deg", getattr(backend_module, "degrees", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Radians")
def _radians(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _radians operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "radians", getattr(backend_module, "deg2rad", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Cbrt")
def _cbrt(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cbrt operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "cbrt", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return backend_module.sign(x) * backend_module.power(backend_module.abs(x), 1.0 / 3.0)


@global_eager_registry.register("Fix")
def _fix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fix", getattr(backend_module, "trunc", None))
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return backend_module.where(x >= 0, backend_module.floor(x), backend_module.ceil(x))


@global_eager_registry.register("Hypot")
def _hypot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _hypot operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "hypot", None)
    if func:
        return func(*args, **kwargs)
    (x, y) = (args[0], args[1])
    return backend_module.sqrt(x**2 + y**2)


@global_eager_registry.register("I0")
def _i0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "i0", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Imag")
def _imag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _imag operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "imag", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Lcm")
def _lcm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _lcm operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "lcm", getattr(backend_module, "least_common_multiple", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Nextafter")
def _nextafter(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _nextafter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import math

    func = getattr(backend_module, "nextafter", getattr(math, "nextafter", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Real")
def _real(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _real operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "real", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Spacing")
def _spacing(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _spacing operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "spacing", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Unwrap")
def _unwrap(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _unwrap operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "unwrap", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Zeta")
def _zeta(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _zeta operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "zeta", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Beta")
def _beta(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _beta operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    kwargs.pop("shape", None)
    kwargs.pop("dtype", None)
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "beta"):
        return backend_module.random.beta(getattr(args[1], "data", args[1]), getattr(args[2], "data", args[2]))
    func = getattr(backend_module, "beta", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Betainc")
def _betainc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _betainc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "betainc", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("AllToAll")
def _all_to_all(backend_module: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _all_to_all operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return tensor


@global_eager_registry.register("ApplyOverAxes")
def _apply_over_axes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _apply_over_axes operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return args[1] if len(args) > 1 else None


@global_eager_registry.register("ArrayRepr")
def _array_repr(backend_module: Any, arr: Any, **kwargs: Any) -> Any:
    """Evaluate _array_repr operation.

    Args:
        backend_module (object): The backend_module parameter.
        arr (object): The arr parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return repr(arr)


@global_eager_registry.register("ArrayStr")
def _array_str(backend_module: Any, arr: Any, **kwargs: Any) -> Any:
    """Evaluate _array_str operation.

    Args:
        backend_module (object): The backend_module parameter.
        arr (object): The arr parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return str(arr)


@global_eager_registry.register("Assign")
def _assign(backend_module: Any, ref: Any, value: Any, **kwargs: Any) -> Any:
    """Evaluate _assign operation.

    Args:
        backend_module (object): The backend_module parameter.
        ref (object): The ref parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return value


@global_eager_registry.register("AssignVariable")
def _assign_variable(backend_module: Any, ref: Any, value: Any, **kwargs: Any) -> Any:
    """Evaluate _assign_variable operation.

    Args:
        backend_module (object): The backend_module parameter.
        ref (object): The ref parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return value


@global_eager_registry.register("AssociativeScan")
def _associative_scan(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _associative_scan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
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
def _atleast_1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atleast_1d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.atleast_1d(*args) if hasattr(backend_module, "atleast_1d") else args[0]


@global_eager_registry.register("Atleast2d")
def _atleast_2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atleast_2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.atleast_2d(*args) if hasattr(backend_module, "atleast_2d") else args[0]


@global_eager_registry.register("Atleast3d")
def _atleast_3d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atleast_3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.atleast_3d(*args) if hasattr(backend_module, "atleast_3d") else args[0]


@global_eager_registry.register("Adjoint")
def _adjoint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _adjoint operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "adjoint", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    if hasattr(backend_module, "conj") and hasattr(backend_module, "transpose"):
        return backend_module.conj(backend_module.transpose(x))
    x_np = backend_module.asarray(x)
    return backend_module.conj(backend_module.transpose(x_np))


@global_eager_registry.register("Bincount")
def _bincount(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bincount operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "bincount", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    weights = kwargs.get("weights", None)
    minlength = kwargs.get("minlength", 0)
    return backend_module.bincount(backend_module.asarray(x), weights=backend_module.asarray(weights) if weights is not None else None, minlength=minlength)


@global_eager_registry.register("Cross")
def _cross(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cross operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "cross", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.cross(backend_module.asarray(args[0]), backend_module.asarray(args[1]), **kwargs)


@global_eager_registry.register("Inner")
def _inner(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _inner operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "inner", None)
    if func:
        return func(*args, **kwargs)
    (a, b) = (args[0], args[1])
    return backend_module.inner(backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("Ediff1d")
def _ediff1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ediff1d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.ediff1d(*args, **kwargs)


@global_eager_registry.register("Gradient")
def _gradient(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _gradient operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.gradient(*args, **kwargs)


@global_eager_registry.register("HardSilu")
def _hardsilu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _hardsilu operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    return x * backend_module.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("Interp")
def _interp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _interp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.interp(*args, **kwargs)


@global_eager_registry.register("Iterable")
def _iterable(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _iterable operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.iterable(*args, **kwargs)


@global_eager_registry.register("Ix")
def _ix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.ix_(*args, **kwargs)


@global_eager_registry.register("Kron")
def _kron(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _kron operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.kron(*args, **kwargs)


@global_eager_registry.register("Outfeed")
def _outfeed(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _outfeed operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "outfeed"):
        return backend_module.lax.outfeed(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("R")
def _r(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _r operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.r_(*args, **kwargs)


@global_eager_registry.register("Rademacher")
def _rademacher(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _rademacher operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", ())
    return backend_module.random.choice([-1, 1], size=shape)


@global_eager_registry.register("Squareplus")
def _squareplus(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _squareplus operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    b = kwargs.get("b", 4.0)
    return 0.5 * (x + backend_module.sqrt(x**2 + b))


@global_eager_registry.register("Trapezoid")
def _trapezoid(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _trapezoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "trapezoid"):
        return backend_module.trapezoid(*args, **kwargs)
    return backend_module.trapz(*args, **kwargs)


@global_eager_registry.register("Vectorize")
def _vectorize(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _vectorize operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.vectorize(*args, **kwargs)


@global_eager_registry.register("IndexInDim")
def _indexindim(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _indexindim operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
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
        slices[axis] = index  # type: ignore
    return x[tuple(slices)]


@global_eager_registry.register("Ppermute")
def _ppermute(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ppermute operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "ppermute"):
        return backend_module.lax.ppermute(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Kronecker")
def _kronecker(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _kronecker operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "kron", getattr(backend_module, "kronecker", None))
    if func:
        return func(*args, **kwargs)
    return backend_module.kron(backend_module.asarray(args[0]), backend_module.asarray(args[1]))


@global_eager_registry.register("Outer")
def _outer(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _outer operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "outer", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.outer(backend_module.asarray(args[0]), backend_module.asarray(args[1]), **kwargs)


@global_eager_registry.register("Fabs")
def _fabs(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fabs operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fabs", getattr(backend_module, "abs", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Gcd")
def _gcd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _gcd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "gcd", getattr(backend_module, "greatest_common_divisor", None))
    if func:
        return func(*args, **kwargs)
    import math

    return math.gcd(*args, **kwargs)


@global_eager_registry.register("Ball")
def _np_ball(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ball operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "ball", getattr(backend_module, "ball", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.zeros(kwargs.get("shape", ()), dtype=np.float32)


@global_eager_registry.register("BetaPdf")
def _np_betapdf(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_betapdf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "betapdf", getattr(backend_module, "betapdf", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.power(args[0], args[1] - 1) * np.power(1 - args[0], args[2] - 1)


@global_eager_registry.register("Gcd")
def _np_gcd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_gcd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "gcd", getattr(backend_module, "gcd", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.gcd(args[0], args[1])


@global_eager_registry.register("Inner")
def _np_inner(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_inner operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "inner", getattr(backend_module, "inner", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.inner(args[0], args[1])


@global_eager_registry.register("R")
def _np_r(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_r operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "r", getattr(backend_module, "r", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.r_[args[0]]


@global_eager_registry.register("Switch")
def _np_switch(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_switch operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "switch", getattr(backend_module, "switch", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[1] if args[0] else args[2]


@global_eager_registry.register("T")
def _np_t(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_t operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "t", getattr(backend_module, "t", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.transpose(args[0])


@global_eager_registry.register("Trapezoid")
def _np_trapezoid(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trapezoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "trapezoid", getattr(backend_module, "trapezoid", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.trapz(*args, **kwargs)


@global_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidalintegral(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trapezoidalintegral operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "trapezoidalintegral", getattr(backend_module, "trapezoidalintegral", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.trapz(*args, **kwargs)


@global_eager_registry.register("Variance")
def _np_variance(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_variance operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "variance", getattr(backend_module, "variance", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.var(*args, **kwargs)


@global_eager_registry.register("Vectorize")
def _np_vectorize(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vectorize operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "vectorize", getattr(backend_module, "vectorize", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.vectorize(*args, **kwargs)


@global_eager_registry.register("Welch")
def _np_welch(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_welch operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "welch", getattr(backend_module, "welch", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.fft.rfft(args[0])


@global_eager_registry.register("WrapKeyData")
def _np_wrapkeydata(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_wrapkeydata operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "wrapkeydata", getattr(backend_module, "wrapkeydata", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]


@global_eager_registry.register("ZeroFraction")
def _np_zerofraction(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_zerofraction operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "zerofraction", getattr(backend_module, "zerofraction", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.sum(args[0] == 0) / np.size(args[0])


@global_eager_registry.register("Zeta")
def _np_zeta(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_zeta operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "zeta", getattr(backend_module, "zeta", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]
