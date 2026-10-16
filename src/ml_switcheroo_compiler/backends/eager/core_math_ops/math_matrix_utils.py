"""math_matrix_utils module."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Callable
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.distributed.host_collectives import np


def _apply_causal_mask(backend_module: EagerValue, scores: EagerValue) -> EagerValue:
    """Apply a causal mask to attention scores.

    Args:
        backend_module: The backend_module parameter.
        scores: The scores parameter.

    Returns:
            object: Result.
    """
    if hasattr(backend_module, "triu") and hasattr(backend_module, "ones") and hasattr(backend_module, "where"):
        causal_mask = getattr(backend_module, "triu")(getattr(backend_module, "ones")(cast(EagerTensorProtocol, scores).shape[-2:]), 1)
        return cast(EagerValue, getattr(backend_module, "where")(causal_mask > 0, float("-inf"), scores))
    return scores


@global_eager_registry.register("Geometric")
def _geometric(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _geometric operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(getattr(backend_module, "random", backend_module), "geometric")(*args, **kwargs))


@global_eager_registry.register("Indices")
def _indices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _indices operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "indices")(*args, **kwargs))


@global_eager_registry.register("MaskIndices")
def _maskindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _maskindices operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(Callable[..., EagerValue], getattr(backend_module, "mask_indices"))(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _tri operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "tri")(*args, **kwargs))


@global_eager_registry.register("Tril")
def _tril(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _tril operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "tril")(*args, **kwargs))


@global_eager_registry.register("TrimZeros")
def _trimzeros(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _trimzeros operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "trim_zeros")(*args, **kwargs))


@global_eager_registry.register("Triu")
def _triu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _triu operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "triu")(*args, **kwargs))


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _fill_diagonal operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "fill_diagonal", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    (x, val) = (args[0], args[1])
    x_np = np.array(x, copy=True)
    np.fill_diagonal(x_np, val, **kwargs)
    return cast(EagerValue, getattr(backend_module, "array")(x_np) if hasattr(backend_module, "array") else x_np)


@global_eager_registry.register("BandPart")
def _np_bandpart(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_bandpart operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.

    Raises:
        ValueError: An exception.
    """
    func = getattr(backend_module, "bandpart", getattr(backend_module, "bandpart", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.triu(np.tril(args[0], getattr(kwargs.get("num_lower", -1), "__int__", lambda: -1)()), -getattr(kwargs.get("num_upper", -1), "__int__", lambda: -1)()))


@global_eager_registry.register("Geometric")
def _np_geometric(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_geometric operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "geometric", getattr(backend_module, "geometric", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.random.geometric(*args, **kwargs))


@global_eager_registry.register("Triangular")
def _np_triangular(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_triangular operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "triangular", getattr(backend_module, "triangular", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.random.triangular(*args, **kwargs))


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_tridiagonal operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "tridiagonal", getattr(backend_module, "tridiagonal", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.diag(args[0]))


@global_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_trilindices operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "trilindices", getattr(backend_module, "trilindices", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.tril_indices(*args, **kwargs))


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_trilindicesfrom operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "trilindicesfrom", getattr(backend_module, "trilindicesfrom", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.tril_indices_from(*args, **kwargs))


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_trimzeros operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "trimzeros", getattr(backend_module, "trimzeros", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.trim_zeros(*args, **kwargs))


@global_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_triuindices operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "triuindices", getattr(backend_module, "triuindices", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.triu_indices(*args, **kwargs))


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_triuindicesfrom operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "triuindicesfrom", getattr(backend_module, "triuindicesfrom", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.triu_indices_from(*args, **kwargs))
