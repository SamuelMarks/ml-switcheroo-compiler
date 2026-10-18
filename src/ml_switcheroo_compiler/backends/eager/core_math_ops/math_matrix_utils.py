"""math_matrix_utils module."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Callable
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)


def _apply_causal_mask(backend_module: EagerValue, scores: EagerValue) -> EagerValue:
    """Apply a causal mask to attention scores.

    Args:
        backend_module: The backend_module parameter.
        scores: The scores parameter.

    Returns:
            EagerValue: Result.
    """
    if hasattr(backend_module, "triu") and hasattr(backend_module, "ones") and hasattr(backend_module, "where"):
        causal_mask = getattr(backend_module, "triu")(getattr(backend_module, "ones")(cast(EagerTensorProtocol, scores).shape[-2:]), 1)
        return cast(EagerValue, getattr(backend_module, "where")(causal_mask > 0, float("-inf"), scores))
    return scores


@global_eager_registry.register("Geometric")
def _geometric(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager geometric operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(getattr(backend_module, "random", backend_module), "geometric")(*args, **kwargs))


@global_eager_registry.register("Indices")
def _indices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager indices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "indices")(*args, **kwargs))


@global_eager_registry.register("MaskIndices")
def _maskindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager maskindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(Callable[..., EagerValue], getattr(backend_module, "mask_indices"))(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tri operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "tri")(*args, **kwargs))


@global_eager_registry.register("Tril")
def _tril(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tril operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "tril")(*args, **kwargs))


@global_eager_registry.register("TrimZeros")
def _trimzeros(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager trimzeros operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "trim_zeros")(*args, **kwargs))


@global_eager_registry.register("Triu")
def _triu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager triu operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "triu")(*args, **kwargs))


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fill diagonal operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fill_diagonal", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    (x, val) = (args[0], args[1])
    x_np = host_backend.array(x, copy=True)
    host_backend.fill_diagonal(x_np, val, **kwargs)
    return cast(EagerValue, getattr(backend_module, "array")(x_np) if hasattr(backend_module, "array") else x_np)


@global_eager_registry.register("BandPart")
def _np_bandpart(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np bandpart operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    func = getattr(backend_module, "bandpart", getattr(backend_module, "bandpart", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.triu(host_backend.tril(args[0], getattr(kwargs.get("num_lower", -1), "__int__", lambda: -1)()), -getattr(kwargs.get("num_upper", -1), "__int__", lambda: -1)()))


@global_eager_registry.register("Geometric")
def _np_geometric(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np geometric operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "geometric", getattr(backend_module, "geometric", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.random.geometric(*args, **kwargs))


@global_eager_registry.register("Triangular")
def _np_triangular(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triangular operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "triangular", getattr(backend_module, "triangular", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.random.triangular(*args, **kwargs))


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonal operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "tridiagonal", getattr(backend_module, "tridiagonal", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.diag(args[0]))


@global_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trilindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "trilindices", getattr(backend_module, "trilindices", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.tril_indices(*args, **kwargs))


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trilindicesfrom operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "trilindicesfrom", getattr(backend_module, "trilindicesfrom", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.tril_indices_from(*args, **kwargs))


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trimzeros operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "trimzeros", getattr(backend_module, "trimzeros", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.trim_zeros(*args, **kwargs))


@global_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triuindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "triuindices", getattr(backend_module, "triuindices", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.triu_indices(*args, **kwargs))


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triuindicesfrom operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "triuindicesfrom", getattr(backend_module, "triuindicesfrom", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.triu_indices_from(*args, **kwargs))
