"""Generic backend-parametric eager reference operations for `reference_math.array_manipulation`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.reference_math.math_scatter import _band_part
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    coerce_opt_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Resize")
def _np_resize(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    shape: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager fallback for _np_resize.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        shape (BackendArg): Input `shape` argument (shape).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    from ml_switcheroo_compiler.backends.eager.vision_ops.resize import resize_bilinear

    x = arr_mod.asarray(x)
    sh_list = [int(coerce_int(v)) for v in coerce_operand_list(arr_mod.asarray(shape).tolist())]
    return resize_bilinear(backend_module, x, (sh_list[0], sh_list[1]))


@global_eager_registry.register("BandPart")
def _np_band_part(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np band part operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    x_val = args[0] if len(args) > 0 else kwargs.get("x")
    lo_val = int(coerce_int(args[1] if len(args) > 1 else kwargs.get("num_lower", -1)))
    hi_val = int(coerce_int(args[2] if len(args) > 2 else kwargs.get("num_upper", -1)))
    return _band_part(x_val, lo_val, hi_val)


@global_eager_registry.register("Diag")
def _np_diag(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_diag.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    kwargs["k"] = kwargs.pop("diagonal", kwargs.pop("k", 0))
    return adapt_backend_module(backend_module).require_symbol("diag")(*args, **kwargs)


@global_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: int = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axis (int): Input `axis` argument (axis).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return [
        adapt_backend_module(backend_module).require_symbol("squeeze")(a, axis=axis)
        for a in adapt_backend_module(backend_module).require_symbol("split")(
            x, adapt_backend_module(x).require_symbol("shape")[axis], axis=axis
        )
    ]


@global_eager_registry.register("Reshape")
def _np_reshape(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np reshape operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape"))
    return adapt_backend_module(backend_module).require_symbol("reshape")(x, shape)


@global_eager_registry.register("Squeeze")
def _np_squeeze(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np squeeze operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    axis: int | None = coerce_opt_int(kwargs.get("axis", args[0] if len(args) > 0 else None))
    return adapt_backend_module(backend_module).require_symbol("squeeze")(x, axis=axis)


@global_eager_registry.register("Transpose")
def _np_transpose(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np transpose operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    axes = kwargs.get("axes", args[0] if len(args) > 0 else None)
    return adapt_backend_module(backend_module).require_symbol("transpose")(x, axes=axes)


@global_eager_registry.register("Rot90")
def _np_rot90(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rot90 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("rot90")(*args, **kwargs)


@global_eager_registry.register("Gather")
def gather_eager(
    np_mod: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager gather implementation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    t = args[0]
    axis: int | None = coerce_opt_int(args[1] if len(args) > 1 else kwargs.get("axis"))
    index = args[2] if len(args) > 2 else kwargs.get("index")
    if adapt_backend_module(t).has_symbol("numpy"):
        t = adapt_backend_module(t).require_symbol("numpy")()
    if adapt_backend_module(index).has_symbol("numpy"):
        index = adapt_backend_module(index).require_symbol("numpy")()
    return adapt_backend_module(np_mod).require_symbol("take_along_axis")(t, index, axis=axis)


@global_eager_registry.register("Stack")
def stack_eager(
    np_mod: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager stack implementation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    raw_tensors = args[0] if len(args) > 0 else (kwargs.get("tensors") or [])
    tensors = (
        coerce_operand_list(raw_tensors)
        if isinstance(raw_tensors, (list, tuple)) or adapt_backend_module(raw_tensors).has_symbol("shape")
        else []
    )
    axis: int = coerce_int(args[1] if len(args) > 1 else kwargs.get("axis", 0))
    arrays: list[BackendArg] = [
        adapt_backend_module(t).require_symbol("numpy")() if adapt_backend_module(t).has_symbol("numpy") else t
        for t in tensors
    ]
    return adapt_backend_module(np_mod).require_symbol("stack")(arrays, axis=axis)


@global_eager_registry.register("Tile")
def _np_tile(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    reps: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np tile operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        reps (BackendArg): Input `reps` argument (reps).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("tile")(x, reps)


@global_eager_registry.register("Permute")
def _np_permute(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np permute operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dims = kwargs.get("dims", args[0] if len(args) > 0 else None)
    return adapt_backend_module(backend_module).require_symbol("transpose")(x, axes=dims)


@global_eager_registry.register("Triu")
def _np_triu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_triu.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return adapt_backend_module(backend_module).require_symbol("triu")(*args, **kwargs)


@global_eager_registry.register("Tril")
def _np_tril(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_tril.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return adapt_backend_module(backend_module).require_symbol("tril")(*args, **kwargs)


@global_eager_registry.register("ExpandDims")
def _np_expand_dims(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np expand dims operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    axis: int | None = coerce_opt_int(args[0] if len(args) > 0 else kwargs.get("axis"))
    return adapt_backend_module(backend_module).require_symbol("expand_dims")(x, axis=axis)


@global_eager_registry.register("Atleast1d")
def _np_atleast_1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np atleast 1d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_1d")(x)


@global_eager_registry.register("Atleast2d")
def _np_atleast_2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np atleast 2d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_2d")(x)


@global_eager_registry.register("Atleast3d")
def _np_atleast_3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np atleast 3d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_3d")(x)


@global_eager_registry.register("Append")
def _np_append(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_append.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("append")(*args, **kwargs)


@global_eager_registry.register("ColumnStack")
def _np_column_stack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_column_stack.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return adapt_backend_module(backend_module).require_symbol("column_stack")(tup)


@global_eager_registry.register("Dsplit")
def _np_dsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_dsplit.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("dsplit")(*args, **kwargs)


@global_eager_registry.register("Dstack")
def _np_dstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_dstack.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return adapt_backend_module(backend_module).require_symbol("dstack")(tup)


@global_eager_registry.register("Hsplit")
def _np_hsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_hsplit.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("hsplit")(*args, **kwargs)


@global_eager_registry.register("Hstack")
def _np_hstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_hstack.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return adapt_backend_module(backend_module).require_symbol("hstack")(tup)


@global_eager_registry.register("Vsplit")
def _np_vsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_vsplit.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("vsplit")(*args, **kwargs)


@global_eager_registry.register("Vstack")
def _np_vstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_vstack.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return adapt_backend_module(backend_module).require_symbol("vstack")(tup)


@global_eager_registry.register("Moveaxis")
def _np_moveaxis(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_moveaxis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("moveaxis")(*args, **kwargs)


@global_eager_registry.register("Swapaxes")
def _np_swapaxes(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_swapaxes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("swapaxes")(*args, **kwargs)


@global_eager_registry.register("Roll")
def _np_roll(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_roll.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("roll")(*args, **kwargs)


@global_eager_registry.register("Block")
def _np_block(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_block.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("block")(*args, **kwargs)


@global_eager_registry.register("Delete")
def _np_delete(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_delete.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("delete")(*args, **kwargs)


@global_eager_registry.register("DiagIndices")
def _np_diag_indices(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_diag_indices.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("diag_indices")(*args, **kwargs)


@global_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_diag_indices_from.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("diag_indices_from")(*args, **kwargs)


@global_eager_registry.register("Diagflat")
def _np_diagflat(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_diagflat.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("diagflat")(*args, **kwargs)


@global_eager_registry.register("FillDiagonal")
def _np_fill_diagonal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_fill_diagonal.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = args[0] if len(args) > 0 else kwargs.get("a")
    val = args[1] if len(args) > 1 else kwargs.get("val")
    wrap: bool = bool(kwargs.get("wrap", False))
    adapt_backend_module(backend_module).require_symbol("fill_diagonal")(a, val, wrap=wrap)
    return a


@global_eager_registry.register("Insert")
def _np_insert(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Eager fallback for _np_insert.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("insert")(*args, **kwargs)
