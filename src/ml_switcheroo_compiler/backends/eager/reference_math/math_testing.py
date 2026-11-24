"""Generic backend-parametric eager reference operations for `reference_math.math_testing`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Piecewise")
def _np_piecewise(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np piecewise operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("piecewise")(
        arr_mod.asarray(args[0]), args[1], args[2], *args[3:], **kwargs
    )


@global_eager_registry.register("PromoteTypes")
def _np_promotetypes(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Return the data type with the smallest size and smallest scalar kind to which both given types can be.

    safely cast.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("promote_types")(args[0], args[1])


@global_eager_registry.register("ApplyAlongAxis")
def _np_apply_along_axis(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Apply a function to 1-D slices along the given axis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("apply_along_axis")(*args, **kwargs)


@global_eager_registry.register("ArrayEquiv")
def _np_array_equiv_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ArrayEquiv via array_equiv.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("array_equiv")(*args, **kwargs)


@global_eager_registry.register("BroadcastArrays")
def _np_broadcast_arrays_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement BroadcastArrays via broadcast_arrays.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("broadcast_arrays")(*args, **kwargs)


@global_eager_registry.register("CanCast")
def _np_can_cast_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement CanCast via can_cast.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("can_cast")(*args, **kwargs)


@global_eager_registry.register("Histogram")
def _np_histogram_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogram via histogram.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    mod = adapt_backend_module(backend_module)
    norm_args: tuple[BackendArg, ...] = (
        (mod.require_symbol("asarray")(args[0], dtype="uint8"), *args[1:])
        if args and mod.has_symbol("asarray") and str(adapt_backend_module(args[0]).get_symbol("dtype", "")) == "bool"
        else args
    )
    return mod.require_symbol("histogram")(*norm_args, **kwargs)


@global_eager_registry.register("Histogram2d")
def _np_histogram2d_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram2d")(*args, **kwargs)


@global_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    mod = adapt_backend_module(backend_module)
    norm_args: tuple[BackendArg, ...] = (
        (mod.require_symbol("asarray")(args[0], dtype="uint8"), *args[1:])
        if args and mod.has_symbol("asarray") and str(adapt_backend_module(args[0]).get_symbol("dtype", "")) == "bool"
        else args
    )
    return mod.require_symbol("histogram_bin_edges")(*norm_args, **kwargs)


@global_eager_registry.register("Histogramdd")
def _np_histogramdd_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogramdd")(*args, **kwargs)


@global_eager_registry.register("Iscomplex")
def _np_iscomplex_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Iscomplex via iscomplex.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("iscomplex")(*args, **kwargs)


@global_eager_registry.register("Iscomplexobj")
def _np_iscomplexobj_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Iscomplexobj via iscomplexobj.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("iscomplexobj")(*args, **kwargs)


@global_eager_registry.register("Isdtype")
def _np_issubdtype_op_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Isdtype via issubdtype.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("issubdtype")(*args, **kwargs)


@global_eager_registry.register("Isreal")
def _np_isreal_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Isreal via isreal.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("isreal")(*args, **kwargs)


@global_eager_registry.register("Isrealobj")
def _np_isrealobj_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Isrealobj via isrealobj.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("isrealobj")(*args, **kwargs)


@global_eager_registry.register("Isscalar")
def _np_isscalar_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Isscalar via isscalar.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("isscalar")(*args, **kwargs)


@global_eager_registry.register("Issubdtype")
def _np_issubdtype_issubdtype_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Issubdtype via issubdtype.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("issubdtype")(*args, **kwargs)


@global_eager_registry.register("ResultType")
def _np_result_type_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ResultType via result_type.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("result_type")(*args, **kwargs)


@global_eager_registry.register("AssertOp")
def _np_assertop(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement AssertOp.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    condition = args[0] if len(args) > 0 else kwargs.get("condition", None)
    if condition is not None:
        assert arr_mod.all(arr_mod.asarray(condition))
    return arr_mod.array([0.0])
