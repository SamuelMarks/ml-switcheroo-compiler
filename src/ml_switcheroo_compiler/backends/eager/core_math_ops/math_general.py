"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_general`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    ShapeProtocol,
    coerce_int,
    coerce_shape,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

from .math_special import _dispatch_special


@global_eager_registry.register("Deg2Rad")
def _deg2rad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager deg2rad operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("deg2rad", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Degrees")
def _degrees(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager degrees operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "degrees", adapt_backend_module(backend_module).get_symbol("rad2deg", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Rad2Deg")
def _rad2deg(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager rad2deg operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "rad2deg", adapt_backend_module(backend_module).get_symbol("degrees", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Radians")
def _radians(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager radians operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "radians", adapt_backend_module(backend_module).get_symbol("deg2rad", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Cbrt")
def _cbrt(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cbrt operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("cbrt", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return adapt_backend_module(mod).require_symbol("sign")(x) * adapt_backend_module(mod).require_symbol("power")(
        adapt_backend_module(mod).require_symbol("abs")(x), 1.0 / 3.0
    )


@global_eager_registry.register("Fix")
def _fix(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fix operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol(
        "fix", adapt_backend_module(backend_module).get_symbol("trunc", None)
    )
    if func:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x = coerce_numeric(args[0])
    return adapt_backend_module(mod).require_symbol("where")(
        x >= 0,
        adapt_backend_module(mod).require_symbol("floor")(x),
        adapt_backend_module(mod).require_symbol("ceil")(x),
    )


@global_eager_registry.register("Hypot")
def _hypot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager hypot operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("hypot", None)
    if func:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    (x, y) = (coerce_numeric(args[0]), coerce_numeric(args[1]))
    return adapt_backend_module(backend_module).require_symbol("sqrt")(x**2 + y**2)


@global_eager_registry.register("I0")
def _i0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager i0 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("i0", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Imag")
def _imag(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager imag operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("imag", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Lcm")
def _lcm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager lcm operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "lcm", adapt_backend_module(backend_module).get_symbol("least_common_multiple", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).lcm(args[0], args[1])


@global_eager_registry.register("Nextafter")
def _nextafter(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager nextafter operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("nextafter", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).nextafter(args[0], args[1])


@global_eager_registry.register("Real")
def _real(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager real operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("real", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).real(args[0])


@global_eager_registry.register("Spacing")
def _spacing(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager spacing operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("spacing", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).spacing(args[0])


@global_eager_registry.register("Unwrap")
def _unwrap(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager unwrap operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("unwrap", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).unwrap(args[0], **kwargs)


@global_eager_registry.register("Beta")
def _beta(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager beta operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    kwargs.pop("shape", None)
    kwargs.pop("dtype", None)
    if adapt_backend_module(backend_module).has_symbol("random") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("random")
    ).has_symbol("beta"):
        return (
            adapt_backend_module(mod)
            .require_symbol("random")
            .beta(
                adapt_backend_module(args[1]).get_symbol("data", args[1]),
                adapt_backend_module(args[2]).get_symbol("data", args[2]),
            )
        )
    return _dispatch_special("Beta", "beta", backend_module, *args, **kwargs)


@global_eager_registry.register("Betainc")
def _betainc(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager betainc operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Betainc", "betainc", backend_module, *args, **kwargs)


@global_eager_registry.register("AllToAll")
def _all_to_all(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager all to all operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return tensor


@global_eager_registry.register("ApplyOverAxes")
def _apply_over_axes(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager apply over axes operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return args[1] if len(args) > 1 else None


@global_eager_registry.register("ArrayRepr")
def _array_repr(
    backend_module: BackendModuleProtocol | types.ModuleType | None, arr: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager array repr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        arr: Input `arr` argument (arr).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return repr(arr)


@global_eager_registry.register("ArrayStr")
def _array_str(
    backend_module: BackendModuleProtocol | types.ModuleType | None, arr: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager array str operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        arr: Input `arr` argument (arr).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return str(arr)


@global_eager_registry.register("Assign")
def _assign(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    ref: BackendArg,
    value: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager assign operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        ref: Input `ref` argument (ref).
        value: Input `value` argument (value).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return value


@global_eager_registry.register("AssignVariable")
def _assign_variable(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    ref: BackendArg,
    value: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager assign variable operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        ref: Input `ref` argument (ref).
        value: Input `value` argument (value).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return value


@global_eager_registry.register("AssociativeScan")
def _associative_scan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager associative scan operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    fn = args[0]
    elems = args[1]
    axis = kwargs.get("axis", 0)
    elems_arr = adapt_backend_module(mod).require_symbol("array")(elems)
    out = adapt_backend_module(mod).require_symbol("empty_like")(elems_arr)
    elems_arr = adapt_backend_module(mod).require_symbol("moveaxis")(elems_arr, axis, 0)
    out = adapt_backend_module(mod).require_symbol("moveaxis")(out, axis, 0)
    acc = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = require_callable(fn)(acc, elems_arr[i])
        out[i] = acc
    out = adapt_backend_module(mod).require_symbol("moveaxis")(out, 0, axis)
    return out


@global_eager_registry.register("Atleast1d")
def _atleast_1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager atleast 1d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return (
        adapt_backend_module(backend_module).require_symbol("atleast_1d")(*args)
        if adapt_backend_module(backend_module).has_symbol("atleast_1d")
        else args[0]
    )


@global_eager_registry.register("Atleast2d")
def _atleast_2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager atleast 2d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return (
        adapt_backend_module(backend_module).require_symbol("atleast_2d")(*args)
        if adapt_backend_module(backend_module).has_symbol("atleast_2d")
        else args[0]
    )


@global_eager_registry.register("Atleast3d")
def _atleast_3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager atleast 3d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return (
        adapt_backend_module(backend_module).require_symbol("atleast_3d")(*args)
        if adapt_backend_module(backend_module).has_symbol("atleast_3d")
        else args[0]
    )


@global_eager_registry.register("Adjoint")
def _adjoint(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager adjoint operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("adjoint", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    if adapt_backend_module(backend_module).has_symbol("conj") and adapt_backend_module(backend_module).has_symbol(
        "transpose"
    ):
        return adapt_backend_module(mod).require_symbol("conj")(
            adapt_backend_module(mod).require_symbol("transpose")(x)
        )
    x_np = adapt_backend_module(mod).require_symbol("asarray")(x)
    fallback_mod = _resolve_eager_mod(backend_module)
    return fallback_mod.conj(fallback_mod.transpose(x_np))


@global_eager_registry.register("Bincount")
def _bincount(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bincount operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("bincount", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    weights = kwargs.get("weights", None)
    minlength = kwargs.get("minlength", 0)
    return adapt_backend_module(mod).require_symbol("bincount")(
        adapt_backend_module(mod).require_symbol("asarray")(x),
        weights=adapt_backend_module(mod).require_symbol("asarray")(weights) if weights is not None else None,
        minlength=minlength,
    )


@global_eager_registry.register("Cross")
def _cross(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cross operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("cross", None)
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(mod).require_symbol("cross")(
        adapt_backend_module(mod).require_symbol("asarray")(args[0]),
        adapt_backend_module(mod).require_symbol("asarray")(args[1]),
        **kwargs,
    )


@global_eager_registry.register("Inner")
def _inner(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager inner operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("inner", None)
    if func:
        return func(*args, **kwargs)
    (a, b) = (args[0], args[1])
    return adapt_backend_module(mod).require_symbol("inner")(
        adapt_backend_module(mod).require_symbol("asarray")(a), adapt_backend_module(mod).require_symbol("asarray")(b)
    )


@global_eager_registry.register("Ediff1d")
def _ediff1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ediff1d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("ediff1d")(*args, **kwargs)


@global_eager_registry.register("Gradient")
def _gradient(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager gradient operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("gradient")(*args, **kwargs)


@global_eager_registry.register("HardSilu")
def _hardsilu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager hardsilu operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x = coerce_numeric(args[0])
    return x * adapt_backend_module(mod).require_symbol("clip")(x + 3, 0, 6) / 6


@global_eager_registry.register("Interp")
def _interp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager interp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("interp")(*args, **kwargs)


@global_eager_registry.register("Iterable")
def _iterable(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager iterable operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("iterable")(*args, **kwargs)


@global_eager_registry.register("Ix")
def _ix(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ix operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("ix_")(*args, **kwargs)


@global_eager_registry.register("Kron")
def _kron(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager kron operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("kron")(*args, **kwargs)


@global_eager_registry.register("Outfeed")
def _outfeed(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager outfeed operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("lax") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("lax")
    ).has_symbol("outfeed"):
        return adapt_backend_module(mod).require_symbol("lax").outfeed(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("R")
def _r(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager r operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("r_")(*args, **kwargs)


@global_eager_registry.register("Rademacher")
def _rademacher(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager rademacher operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    shape = kwargs.get("shape", ())
    return adapt_backend_module(backend_module).require_symbol("random").choice([-1, 1], size=shape)


@global_eager_registry.register("Squareplus")
def _squareplus(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager squareplus operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float, coerce_numeric

    x = coerce_numeric(args[0])
    b = coerce_float(kwargs.get("b", 4.0))
    return 0.5 * (x + adapt_backend_module(mod).require_symbol("sqrt")(x**2 + b))


@global_eager_registry.register("Trapezoid")
def _trapezoid(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager trapezoid operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("trapezoid"):
        return adapt_backend_module(mod).require_symbol("trapezoid")(*args, **kwargs)
    return adapt_backend_module(mod).require_symbol("trapz")(*args, **kwargs)


@global_eager_registry.register("Vectorize")
def _vectorize(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager vectorize operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("vectorize")(*args, **kwargs)


@global_eager_registry.register("IndexInDim")
def _indexindim(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager indexindim operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    if not adapt_backend_module(backend_module).has_symbol("array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    index = args[1] if len(args) > 1 else kwargs.get("index")
    axis = int(coerce_int(kwargs.get("axis", 0)))
    keepdims = kwargs.get("keepdims", False)
    raw_x: BackendArg = x
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    x_arr = coerce_array(backend_module, x)
    x_shape = coerce_shape(raw_x.shape) if isinstance(raw_x, ShapeProtocol) else ()
    slices: list[BackendArg] = [slice(None)] * len(x_shape)
    if keepdims and isinstance(index, int):
        slices[axis] = slice(index, index + 1)
    else:
        slices[axis] = index
    return x_arr[tuple(slices)]


@global_eager_registry.register("Ppermute")
def _ppermute(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ppermute operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("lax") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("lax")
    ).has_symbol("ppermute"):
        return adapt_backend_module(mod).require_symbol("lax").ppermute(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Kronecker")
def _kronecker(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager kronecker operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol(
        "kron", adapt_backend_module(backend_module).get_symbol("kronecker", None)
    )
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(mod).require_symbol("kron")(
        adapt_backend_module(mod).require_symbol("asarray")(args[0]),
        adapt_backend_module(mod).require_symbol("asarray")(args[1]),
    )


@global_eager_registry.register("Outer")
def _outer(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager outer operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("outer", None)
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(mod).require_symbol("outer")(
        adapt_backend_module(mod).require_symbol("asarray")(args[0]),
        adapt_backend_module(mod).require_symbol("asarray")(args[1]),
        **kwargs,
    )


@global_eager_registry.register("Fabs")
def _fabs(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fabs operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fabs", adapt_backend_module(backend_module).get_symbol("abs", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Gcd")
def _gcd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager gcd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "gcd", adapt_backend_module(backend_module).get_symbol("greatest_common_divisor", None)
    )
    if func:
        return func(*args, **kwargs)
    import math

    return math.gcd(*(int(coerce_int(a)) for a in args))


@global_eager_registry.register("Ball")
def _np_ball(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ball operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("ball", None)
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.zeros(kwargs.get("shape", ()), dtype=mod.float32)


@global_eager_registry.register("BetaPdf")
def _np_betapdf(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np betapdf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BetaPdf", "betapdf", backend_module, *args, **kwargs)


@global_eager_registry.register("Gcd")
def _np_gcd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np gcd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("gcd", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).gcd(args[0], args[1])


@global_eager_registry.register("Inner")
def _np_inner(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np inner operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("inner", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).inner(args[0], args[1])


@global_eager_registry.register("R")
def _np_r(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np r operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("r", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).r_[args[0]]


@global_eager_registry.register("Switch")
def _np_switch(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np switch operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("switch", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[1] if args[0] else args[2]


@global_eager_registry.register("T")
def _np_t(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np t operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("t", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).transpose(args[0])


@global_eager_registry.register("Trapezoid")
def _np_trapezoid(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trapezoid operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "trapezoid", adapt_backend_module(backend_module).get_symbol("trapz", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    trapz_fn = adapt_backend_module(mod).get_symbol("trapezoid", adapt_backend_module(mod).get_symbol("trapz", None))
    return require_callable(trapz_fn)(*args, **kwargs)


@global_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidalintegral(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trapezoidalintegral operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "trapezoidalintegral",
        adapt_backend_module(backend_module).get_symbol(
            "trapezoid", adapt_backend_module(backend_module).get_symbol("trapz", None)
        ),
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    trapz_fn = adapt_backend_module(mod).get_symbol("trapezoid", adapt_backend_module(mod).get_symbol("trapz", None))
    return require_callable(trapz_fn)(*args, **kwargs)


@global_eager_registry.register("Variance")
def _np_variance(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np variance operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("variance", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).var(*args, **kwargs)


@global_eager_registry.register("Vectorize")
def _np_vectorize(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vectorize operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("vectorize", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).vectorize(*args, **kwargs)


@global_eager_registry.register("Welch")
def _np_welch(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np welch operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("welch", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).fft.rfft(args[0])


@global_eager_registry.register("WrapKeyData")
def _np_wrapkeydata(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np wrapkeydata operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("wrapkeydata", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("ZeroFraction")
def _np_zerofraction(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np zerofraction operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("zerofraction", None)
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.sum(args[0] == 0) / mod.size(args[0])


@global_eager_registry.register("Zeta")
def _np_zeta(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np zeta operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Zeta", "zeta", backend_module, *args, **kwargs)
