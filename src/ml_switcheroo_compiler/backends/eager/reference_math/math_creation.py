"""Generic backend-parametric eager reference operations for `reference_math.math_creation`."""

from __future__ import annotations

import types
import typing

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.linalg.linear_operator import (
        LinearOperatorIdentity,
        LinearOperatorScaledIdentity,
        LinearOperatorZeros,
    )


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Fromfunction")
def _np_fromfunction_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> EagerTensorProtocol:
    """Construct an array by executing a function over each coordinate.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (BackendArg): Keyword arguments.

    Returns:
        EagerTensorProtocol: The computed array from function coordinates.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("fromfunction")(*args, **kwargs))


@global_eager_registry.register("Fromiter")
def _np_fromiter_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> EagerTensorProtocol:
    """Create a new 1-dimensional array from an iterable object.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (BackendArg): Keyword arguments.

    Returns:
        EagerTensorProtocol: Output 1-dimensional array.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("fromiter")(*args, **kwargs))


@global_eager_registry.register("Frompyfunc")
def _np_frompyfunc_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> BackendArg:
    """Takes an arbitrary Python function and returns a NumPy universal function.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        BackendArg: Universal function wrapper.
    """
    return adapt_backend_module(backend_module).require_symbol("frompyfunc")(*args, **kwargs)


@global_eager_registry.register("Geomspace")
def _np_geomspace_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | bool | str | None,
) -> EagerTensorProtocol:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (start, stop, num).
        **kwargs (float | int | bool | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Computed geometric progression array.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("geomspace")(*args, **kwargs))


@global_eager_registry.register("Zeros")
def _np_zeros_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Array of zeros with given shape.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("zeros")(*args, **kwargs))


@global_eager_registry.register("Ones")
def _np_ones_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return a new array of given shape and type, filled with ones.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Array of ones with given shape.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("ones")(*args, **kwargs))


@global_eager_registry.register("Empty")
def _np_empty_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Uninitialized array with given shape.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("empty")(*args, **kwargs))


@global_eager_registry.register("Full")
def _np_full_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Array filled with constant value.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("full")(*args, **kwargs))


@global_eager_registry.register("ZerosLike")
def _np_zeros_like_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Array of zeros like prototype.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("zeros_like")(*args, **kwargs))


@global_eager_registry.register("OnesLike")
def _np_ones_like_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Array of ones like prototype.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("ones_like")(*args, **kwargs))


@global_eager_registry.register("EmptyLike")
def _np_empty_like_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return an empty array with the same shape and type as a given array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Empty array like prototype.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("empty_like")(*args, **kwargs))


@global_eager_registry.register("FullLike")
def _np_full_like_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | None,
) -> EagerTensorProtocol:
    """Return a full array with the same shape and type as a given array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (str | str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Full array like prototype filled with value.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("full_like")(*args, **kwargs))


@global_eager_registry.register("Arange")
def _np_arange_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | None,
) -> EagerTensorProtocol:
    """Return evenly spaced values within a given interval.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (start, stop, step).
        **kwargs (str | None): Keyword arguments.

    Returns:
        EagerTensorProtocol: Evenly spaced values array.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("arange")(*args, **kwargs))


@global_eager_registry.register("LinearOperatorIdentity")
def _np_linearoperatoridentity(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | str | None,
) -> LinearOperatorIdentity:
    """Implement LinearOperatorIdentity creation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments for LinearOperatorIdentity.
        **kwargs (int | str | str | None): Keyword arguments.

    Returns:
        LinearOperatorIdentity: Initialized linear operator identity instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorIdentity

    return LinearOperatorIdentity(*args, **kwargs)


@global_eager_registry.register("LinearOperatorScaledIdentity")
def _np_linearoperatorscaledidentity(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | EagerTensorProtocol | str | str | None,
) -> LinearOperatorScaledIdentity:
    """Implement LinearOperatorScaledIdentity creation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (int | float | EagerTensorProtocol | str | str | None):
            Keyword arguments.

    Returns:
        LinearOperatorScaledIdentity: Initialized linear operator scaled identity instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorScaledIdentity

    return LinearOperatorScaledIdentity(*args, **kwargs)


@global_eager_registry.register("LinearOperatorZeros")
def _np_linearoperatorzeros(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | str | None,
) -> LinearOperatorZeros:
    """Implement LinearOperatorZeros creation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments for LinearOperatorZeros.
        **kwargs (int | str | str | None): Keyword arguments.

    Returns:
        LinearOperatorZeros: Initialized linear operator zeros instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorZeros

    return LinearOperatorZeros(*args, **kwargs)


@global_eager_registry.register("FromDlpack")
def _np_fromdlpack(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | str | None,
) -> EagerTensorProtocol | int | float:
    """Evaluate from_dlpack conversion or return object fallback.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (dlpack capsule/tensor).
        **kwargs (Union[int, float, str, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, int, float]: Tensor converted from DLPack.
    """
    if adapt_backend_module(backend_module).has_symbol("from_dlpack"):
        return _ensure_arr(adapt_backend_module(backend_module).require_symbol("from_dlpack")(*args, **kwargs))
    return _ensure_arr(args[0])


@global_eager_registry.register("Frombuffer")
def _np_frombuffer(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> EagerTensorProtocol | None:
    """Interpret a buffer as a 1-dimensional array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (buffer, dtype, count, offset).
        **kwargs (BackendArg): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, None]: 1-D array from buffer or None if no arguments provided.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if not args:
        return None
    return _ensure_arr(adapt_backend_module(arr_mod).require_symbol("frombuffer")(args[0], **kwargs))
