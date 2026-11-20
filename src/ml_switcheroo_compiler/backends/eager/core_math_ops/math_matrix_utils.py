"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix_utils`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _apply_causal_mask(
    backend_module: BackendModuleProtocol | types.ModuleType | None, scores: BackendArg
) -> BackendArg:
    """Apply a causal mask to attention scores.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        scores: Input `scores` argument (scores).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if (
        adapt_backend_module(backend_module).has_symbol("triu")
        and adapt_backend_module(backend_module).has_symbol("ones")
        and adapt_backend_module(backend_module).has_symbol("where")
    ):
        causal_mask = adapt_backend_module(backend_module).require_symbol("triu")(
            adapt_backend_module(backend_module).require_symbol("ones")(
                coerce_shape(adapt_backend_module(scores).require_symbol("shape"))[-2:]
            ),
            1,
        )
        return adapt_backend_module(backend_module).require_symbol("where")(causal_mask > 0, float("-inf"), scores)
    return scores


@global_eager_registry.register("Geometric")
def _geometric(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager geometric operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(
        adapt_backend_module(backend_module).get_symbol("random", backend_module)
    ).require_symbol("geometric")(*args, **kwargs)


@global_eager_registry.register("Indices")
def _indices(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager indices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("indices")(*args, **kwargs)


@global_eager_registry.register("MaskIndices")
def _maskindices(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager maskindices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_op("mask_indices")(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager tri operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("tri")(*args, **kwargs)


@global_eager_registry.register("Tril")
def _tril(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager tril operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("tril")(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _trimzeros(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager trimzeros operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("trim_zeros")(*args, **kwargs)


@global_eager_registry.register("Triu")
def _triu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager triu operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("triu")(*args, **kwargs)


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fill diagonal operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("fill_diagonal", None)
    if func:
        return func(*args, **kwargs)
    (x, val) = (args[0], args[1])
    mod = _resolve_eager_mod(backend_module)
    x_np = mod.array(x, copy=True)
    mod.fill_diagonal(x_np, val, **kwargs)
    return (
        adapt_backend_module(backend_module).require_symbol("array")(x_np)
        if adapt_backend_module(backend_module).has_symbol("array")
        else x_np
    )


@global_eager_registry.register("BandPart")
def _np_bandpart(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np bandpart operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    func = adapt_backend_module(backend_module).get_symbol("bandpart", None)
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.triu(
        mod.tril(args[0], int(coerce_int(kwargs.get("num_lower", -1)))), -int(coerce_int(kwargs.get("num_upper", -1)))
    )


@global_eager_registry.register("Geometric")
def _np_geometric(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np geometric operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("geometric", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).random.geometric(*args, **kwargs)


@global_eager_registry.register("Triangular")
def _np_triangular(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np triangular operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("triangular", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).random.triangular(*args, **kwargs)


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tridiagonal operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("tridiagonal", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).diag(args[0])


@global_eager_registry.register("TrilIndices")
def _np_trilindices(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trilindices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("trilindices", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).tril_indices(*args, **kwargs)


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trilindicesfrom operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("trilindicesfrom", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).tril_indices_from(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trimzeros operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("trimzeros", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).trim_zeros(*args, **kwargs)


@global_eager_registry.register("TriuIndices")
def _np_triuindices(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np triuindices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("triuindices", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).triu_indices(*args, **kwargs)


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np triuindicesfrom operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("triuindicesfrom", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).triu_indices_from(*args, **kwargs)
