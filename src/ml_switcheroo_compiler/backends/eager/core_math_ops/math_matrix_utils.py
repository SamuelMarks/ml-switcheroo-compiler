"""Implementation of math matrix utils operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix_utils`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    as_callable,
    as_tensor,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.shape import _get_attr, _has_attr


def _apply_causal_mask(backend_module: BackendModuleProtocol, scores: EagerValue) -> EagerValue:
    """Apply a causal mask to attention scores.

    Args:
        backend_module: The backend_module parameter.
        scores: The scores parameter.

    Returns:
            EagerValue: Result.
    """
    if _has_attr(backend_module, "triu") and _has_attr(backend_module, "ones") and _has_attr(backend_module, "where"):
        causal_mask = _get_attr(backend_module, "triu")(_get_attr(backend_module, "ones")(as_tensor(scores).shape[-2:]), 1)
        return as_tensor(_get_attr(backend_module, "where")(causal_mask > 0, float("-inf"), scores))
    return scores


@global_eager_registry.register("Geometric")
def _geometric(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager geometric operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(_get_attr(backend_module, "random", backend_module), "geometric")(*args, **kwargs))


@global_eager_registry.register("Indices")
def _indices(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager indices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "indices")(*args, **kwargs))


@global_eager_registry.register("MaskIndices")
def _maskindices(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager maskindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_callable(_get_attr(backend_module, "mask_indices"))(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tri operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "tri")(*args, **kwargs))


@global_eager_registry.register("Tril")
def _tril(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tril operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "tril")(*args, **kwargs))


@global_eager_registry.register("TrimZeros")
def _trimzeros(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager trimzeros operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "trim_zeros")(*args, **kwargs))


@global_eager_registry.register("Triu")
def _triu(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager triu operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(_get_attr(backend_module, "triu")(*args, **kwargs))


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fill diagonal operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "fill_diagonal", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    (x, val) = (args[0], args[1])
    mod = _resolve_eager_mod(backend_module)
    x_np = mod.array(x, copy=True)
    mod.fill_diagonal(x_np, val, **kwargs)
    return _get_attr(backend_module, "array")(x_np) if _has_attr(backend_module, "array") else x_np


@global_eager_registry.register("BandPart")
def _np_bandpart(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    func = _get_attr(backend_module, "bandpart", _get_attr(backend_module, "bandpart", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.triu(mod.tril(args[0], _get_attr(kwargs.get("num_lower", -1), "__int__", lambda: -1)()), -_get_attr(kwargs.get("num_upper", -1), "__int__", lambda: -1)())


@global_eager_registry.register("Geometric")
def _np_geometric(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np geometric operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "geometric", _get_attr(backend_module, "geometric", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.geometric(*args, **kwargs)


@global_eager_registry.register("Triangular")
def _np_triangular(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triangular operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "triangular", _get_attr(backend_module, "triangular", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.triangular(*args, **kwargs)


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonal operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "tridiagonal", _get_attr(backend_module, "tridiagonal", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).diag(args[0])


@global_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trilindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "trilindices", _get_attr(backend_module, "trilindices", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).tril_indices(*args, **kwargs)


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trilindicesfrom operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "trilindicesfrom", _get_attr(backend_module, "trilindicesfrom", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).tril_indices_from(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trimzeros operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "trimzeros", _get_attr(backend_module, "trimzeros", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).trim_zeros(*args, **kwargs)


@global_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triuindices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "triuindices", _get_attr(backend_module, "triuindices", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).triu_indices(*args, **kwargs)


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triuindicesfrom operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = _get_attr(backend_module, "triuindicesfrom", _get_attr(backend_module, "triuindicesfrom", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).triu_indices_from(*args, **kwargs)
