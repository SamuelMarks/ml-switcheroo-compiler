"""Generic backend-parametric eager reference operations for `reference_math.math_manipulation`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
    adapt_backend_module,
)


@global_eager_registry.register("Append")
def _np_append(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Append values to the end of an array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("append")(*args, **kwargs)


@global_eager_registry.register("ArgWhere")
def _np_argwhere(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("argwhere")(*args, **kwargs)


@global_eager_registry.register("Choose")
def _np_choose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Construct an array from an index array and a list of arrays to choose from.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("choose")(*args, **kwargs)


@global_eager_registry.register("ColumnStack")
def _np_column_stack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("column_stack")(*args, **kwargs)


@global_eager_registry.register("Compress")
def _np_compress(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Return selected slices of an array along given axis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("compress")(*args, **kwargs)


@global_eager_registry.register("Delete")
def _np_delete_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Delete via delete.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("delete")(*args, **kwargs)


@global_eager_registry.register("Extract")
def _np_extract_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Extract via extract.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("extract")(*args, **kwargs)


@global_eager_registry.register("Flatnonzero")
def _np_flatnonzero_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Flatnonzero via flatnonzero.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("flatnonzero")(*args, **kwargs)


@global_eager_registry.register("Flip")
def _np_flip_op_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Flip via flip.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("flip")(*args, **kwargs)


@global_eager_registry.register("Fliplr")
def _np_fliplr_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Fliplr via fliplr.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("fliplr")(*args, **kwargs)


@global_eager_registry.register("Flipud")
def _np_flipud_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Flipud via flipud.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("flipud")(*args, **kwargs)


@global_eager_registry.register("Reverse")
def _np_flip_reverse_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Reverse via flip.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("flip")(*args, **kwargs)


@global_eager_registry.register("Insert")
def _np_insert_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Insert via insert.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("insert")(*args, **kwargs)


@global_eager_registry.register("Nonzero")
def _np_nonzero_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Nonzero via nonzero.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    mod = adapt_backend_module(backend_module)
    norm_args: tuple[BackendArg, ...] = (
        (mod.require_symbol("atleast_1d")(args[0]), *args[1:])
        if args and mod.has_symbol("atleast_1d") and adapt_backend_module(args[0]).get_symbol("ndim", 0) == 0
        else args
    )
    return mod.require_symbol("nonzero")(*norm_args, **kwargs)


@global_eager_registry.register("Resize")
def _np_resize_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Resize via resize.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("resize")(*args, **kwargs)


@global_eager_registry.register("ExtractPatchesOptions")
def _np_extractpatchesoptions(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ExtractPatchesOptions.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.vision.bbox import ExtractPatchesOptions

    return ExtractPatchesOptions(*args, **kwargs)


@global_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorArrayStack.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    handle = args[0]
    return adapt_backend_module(backend_module).require_symbol("stack")(handle)
