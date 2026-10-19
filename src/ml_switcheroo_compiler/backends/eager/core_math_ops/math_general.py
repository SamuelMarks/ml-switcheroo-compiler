"""Implementation of math general operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_general`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)

from .math_special import _dispatch_special


@global_eager_registry.register("Deg2Rad")
def _deg2rad(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager deg2rad operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "deg2rad", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Degrees")
def _degrees(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager degrees operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "degrees", getattr(backend_module, "rad2deg", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Rad2Deg")
def _rad2deg(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager rad2deg operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "rad2deg", getattr(backend_module, "degrees", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Radians")
def _radians(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager radians operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "radians", getattr(backend_module, "deg2rad", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Cbrt")
def _cbrt(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cbrt operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "cbrt", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, mod.sign(x) * mod.power(mod.abs(x), 1.0 / 3.0))


@global_eager_registry.register("Fix")
def _fix(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fix operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "fix", getattr(backend_module, "trunc", None))
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = cast(EagerTensorProtocol, args[0])
    return cast(EagerValue, mod.where(x >= 0, mod.floor(x), mod.ceil(x)))


@global_eager_registry.register("Hypot")
def _hypot(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager hypot operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "hypot", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    (x, y) = (cast(EagerTensorProtocol, args[0]), cast(EagerTensorProtocol, args[1]))
    return cast(EagerValue, cast(ModuleType, backend_module).sqrt(x**2 + y**2))


@global_eager_registry.register("I0")
def _i0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager i0 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "i0", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Imag")
def _imag(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager imag operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "imag", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Lcm")
def _lcm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager lcm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "lcm", getattr(backend_module, "least_common_multiple", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).lcm(args[0], args[1]))


@global_eager_registry.register("Nextafter")
def _nextafter(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager nextafter operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "nextafter", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).nextafter(args[0], args[1]))


@global_eager_registry.register("Real")
def _real(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager real operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "real", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).real(args[0]))


@global_eager_registry.register("Spacing")
def _spacing(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager spacing operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "spacing", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).spacing(args[0]))


@global_eager_registry.register("Unwrap")
def _unwrap(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager unwrap operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "unwrap", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).unwrap(args[0], **kwargs))


@global_eager_registry.register("Beta")
def _beta(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager beta operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    kwargs.pop("shape", None)
    kwargs.pop("dtype", None)
    if hasattr(backend_module, "random") and hasattr(mod.random, "beta"):
        return cast(EagerValue, mod.random.beta(getattr(args[1], "data", args[1]), getattr(args[2], "data", args[2])))
    return _dispatch_special("Beta", "beta", backend_module, *args, **kwargs)


@global_eager_registry.register("Betainc")
def _betainc(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager betainc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Betainc", "betainc", backend_module, *args, **kwargs)


@global_eager_registry.register("AllToAll")
def _all_to_all(backend_module: EagerValue, tensor: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager all to all operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return tensor


@global_eager_registry.register("ApplyOverAxes")
def _apply_over_axes(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager apply over axes operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return args[1] if len(args) > 1 else None


@global_eager_registry.register("ArrayRepr")
def _array_repr(backend_module: EagerValue, arr: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager array repr operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        arr: The arr parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return repr(arr)


@global_eager_registry.register("ArrayStr")
def _array_str(backend_module: EagerValue, arr: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager array str operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        arr: The arr parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return str(arr)


@global_eager_registry.register("Assign")
def _assign(backend_module: EagerValue, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager assign operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return value


@global_eager_registry.register("AssignVariable")
def _assign_variable(backend_module: EagerValue, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager assign variable operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return value


@global_eager_registry.register("AssociativeScan")
def _associative_scan(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager associative scan operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    fn = cast(EagerTensorProtocol, args[0])
    elems = args[1]
    axis = kwargs.get("axis", 0)
    elems_arr = mod.array(elems)
    out = mod.empty_like(elems_arr)
    elems_arr = mod.moveaxis(elems_arr, axis, 0)
    out = mod.moveaxis(out, axis, 0)
    acc = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = fn.__call__(acc, elems_arr[i])
        out[i] = acc
    out = mod.moveaxis(out, 0, axis)
    return cast(EagerValue, out)


@global_eager_registry.register("Atleast1d")
def _atleast_1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atleast 1d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(ModuleType, backend_module).atleast_1d(*args) if hasattr(backend_module, "atleast_1d") else args[0]


@global_eager_registry.register("Atleast2d")
def _atleast_2d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atleast 2d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(ModuleType, backend_module).atleast_2d(*args) if hasattr(backend_module, "atleast_2d") else args[0]


@global_eager_registry.register("Atleast3d")
def _atleast_3d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atleast 3d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(ModuleType, backend_module).atleast_3d(*args) if hasattr(backend_module, "atleast_3d") else args[0]


@global_eager_registry.register("Adjoint")
def _adjoint(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adjoint operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "adjoint", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    if hasattr(backend_module, "conj") and hasattr(backend_module, "transpose"):
        return cast(EagerValue, mod.conj(mod.transpose(x)))
    x_np = mod.asarray(x)
    fallback_mod = _resolve_eager_mod(backend_module)
    return cast(EagerValue, fallback_mod.conj(fallback_mod.transpose(x_np)))


@global_eager_registry.register("Bincount")
def _bincount(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bincount operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "bincount", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    weights = kwargs.get("weights", None)
    minlength = kwargs.get("minlength", 0)
    return cast(EagerValue, mod.bincount(mod.asarray(x), weights=mod.asarray(weights) if weights is not None else None, minlength=minlength))


@global_eager_registry.register("Cross")
def _cross(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cross operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "cross", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, mod.cross(mod.asarray(args[0]), mod.asarray(args[1]), **kwargs))


@global_eager_registry.register("Inner")
def _inner(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager inner operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "inner", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    (a, b) = (args[0], args[1])
    return cast(EagerValue, mod.inner(mod.asarray(a), mod.asarray(b)))


@global_eager_registry.register("Ediff1d")
def _ediff1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ediff1d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).ediff1d(*args, **kwargs))


@global_eager_registry.register("Gradient")
def _gradient(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager gradient operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).gradient(*args, **kwargs))


@global_eager_registry.register("HardSilu")
def _hardsilu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager hardsilu operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    return cast(EagerValue, x * mod.clip(x + 3, 0, 6) / 6)


@global_eager_registry.register("Interp")
def _interp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager interp operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).interp(*args, **kwargs))


@global_eager_registry.register("Iterable")
def _iterable(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager iterable operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).iterable(*args, **kwargs))


@global_eager_registry.register("Ix")
def _ix(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ix operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).ix_(*args, **kwargs))


@global_eager_registry.register("Kron")
def _kron(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager kron operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).kron(*args, **kwargs))


@global_eager_registry.register("Outfeed")
def _outfeed(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager outfeed operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    if hasattr(backend_module, "lax") and hasattr(mod.lax, "outfeed"):
        return cast(EagerValue, mod.lax.outfeed(*args, **kwargs))
    return args[0] if args else None


@global_eager_registry.register("R")
def _r(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager r operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).r_(*args, **kwargs))


@global_eager_registry.register("Rademacher")
def _rademacher(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager rademacher operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    shape = kwargs.get("shape", ())
    return cast(EagerValue, cast(ModuleType, backend_module).random.choice([-1, 1], size=shape))


@global_eager_registry.register("Squareplus")
def _squareplus(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager squareplus operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    b = kwargs.get("b", 4.0)
    return cast(EagerValue, 0.5 * (x + mod.sqrt(x**2 + cast(EagerTensorProtocol, b))))


@global_eager_registry.register("Trapezoid")
def _trapezoid(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager trapezoid operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    if hasattr(backend_module, "trapezoid"):
        return cast(EagerValue, mod.trapezoid(*args, **kwargs))
    return cast(EagerValue, mod.trapz(*args, **kwargs))


@global_eager_registry.register("Vectorize")
def _vectorize(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager vectorize operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).vectorize(*args, **kwargs))


@global_eager_registry.register("IndexInDim")
def _indexindim(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager indexindim operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not hasattr(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    index = args[1] if len(args) > 1 else kwargs.get("index")
    axis = int(cast(int, kwargs.get("axis", 0)))
    keepdims = kwargs.get("keepdims", False)
    slices: list[EagerValue] = [slice(None)] * len(getattr(x, "shape", ()))
    if keepdims and isinstance(index, int):
        slices[axis] = slice(index, index + 1)
    else:
        slices[axis] = index
    return typing.cast(EagerTensorProtocol, x)[tuple(slices)]


@global_eager_registry.register("Ppermute")
def _ppermute(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ppermute operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    if hasattr(backend_module, "lax") and hasattr(mod.lax, "ppermute"):
        return cast(EagerValue, mod.lax.ppermute(*args, **kwargs))
    return args[0] if args else None


@global_eager_registry.register("Kronecker")
def _kronecker(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager kronecker operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "kron", getattr(backend_module, "kronecker", None))
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, mod.kron(mod.asarray(args[0]), mod.asarray(args[1])))


@global_eager_registry.register("Outer")
def _outer(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager outer operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "outer", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, mod.outer(mod.asarray(args[0]), mod.asarray(args[1]), **kwargs))


@global_eager_registry.register("Fabs")
def _fabs(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fabs operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fabs", getattr(backend_module, "abs", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Gcd")
def _gcd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager gcd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "gcd", getattr(backend_module, "greatest_common_divisor", None))
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    import math

    return cast(EagerValue, getattr(math, "gcd")(*args, **kwargs))


@global_eager_registry.register("Ball")
def _np_ball(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ball operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "ball", getattr(backend_module, "ball", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return cast(EagerValue, mod.zeros(kwargs.get("shape", ()), dtype=mod.float32))


@global_eager_registry.register("BetaPdf")
def _np_betapdf(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np betapdf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BetaPdf", "betapdf", backend_module, *args, **kwargs)


@global_eager_registry.register("Gcd")
def _np_gcd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np gcd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "gcd", getattr(backend_module, "gcd", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).gcd(args[0], args[1]))


@global_eager_registry.register("Inner")
def _np_inner(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np inner operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "inner", getattr(backend_module, "inner", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).inner(args[0], args[1]))


@global_eager_registry.register("R")
def _np_r(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np r operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "r", getattr(backend_module, "r", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).r_[args[0]])


@global_eager_registry.register("Switch")
def _np_switch(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np switch operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "switch", getattr(backend_module, "switch", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[1] if args[0] else args[2]


@global_eager_registry.register("T")
def _np_t(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np t operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "t", getattr(backend_module, "t", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).transpose(args[0]))


@global_eager_registry.register("Trapezoid")
def _np_trapezoid(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trapezoid operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "trapezoid", getattr(backend_module, "trapz", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    trapz_fn = typing.cast("typing.Callable[..., EagerValue]", getattr(mod, "trapezoid", getattr(mod, "trapz", None)))
    return trapz_fn(*args, **kwargs)


@global_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidalintegral(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trapezoidalintegral operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "trapezoidalintegral", getattr(backend_module, "trapezoid", getattr(backend_module, "trapz", None)))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    trapz_fn = typing.cast("typing.Callable[..., EagerValue]", getattr(mod, "trapezoid", getattr(mod, "trapz", None)))
    return trapz_fn(*args, **kwargs)


@global_eager_registry.register("Variance")
def _np_variance(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np variance operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "variance", getattr(backend_module, "variance", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).var(*args, **kwargs))


@global_eager_registry.register("Vectorize")
def _np_vectorize(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vectorize operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "vectorize", getattr(backend_module, "vectorize", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).vectorize(*args, **kwargs))


@global_eager_registry.register("Welch")
def _np_welch(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np welch operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "welch", getattr(backend_module, "welch", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).fft.rfft(args[0]))


@global_eager_registry.register("WrapKeyData")
def _np_wrapkeydata(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np wrapkeydata operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "wrapkeydata", getattr(backend_module, "wrapkeydata", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("ZeroFraction")
def _np_zerofraction(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np zerofraction operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "zerofraction", getattr(backend_module, "zerofraction", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return cast(EagerValue, mod.sum(args[0] == 0) / mod.size(args[0]))


@global_eager_registry.register("Zeta")
def _np_zeta(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np zeta operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Zeta", "zeta", backend_module, *args, **kwargs)
