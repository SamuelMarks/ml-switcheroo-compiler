# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_matrix_utils module."""

from __future__ import annotations

import typing
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def _apply_causal_mask(backend_module: Any, scores: Any) -> Any:
    """Apply a causal mask to attention scores.

    Args:
        backend_module (Any): The backend_module parameter.
        scores (Any): The scores parameter.

    Returns:
            Any: Result.
    """
    if hasattr(backend_module, "triu") and hasattr(backend_module, "ones") and hasattr(backend_module, "where"):
        causal_mask = backend_module.triu(backend_module.ones(scores.shape[-2:]), 1)
        return backend_module.where(causal_mask > 0, float("-inf"), scores)
    return scores


@global_eager_registry.register("Geometric")
def _geometric(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _geometric operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return getattr(backend_module, "random", backend_module).geometric(*args, **kwargs)


@global_eager_registry.register("Indices")
def _indices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _indices operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.indices(*args, **kwargs)


@global_eager_registry.register("MaskIndices")
def _maskindices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _maskindices operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.mask_indices(*args, **kwargs)


@global_eager_registry.register("Tri")
def _tri(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _tri operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.tri(*args, **kwargs)


@global_eager_registry.register("Tril")
def _tril(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _tril operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.tril(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _trimzeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _trimzeros operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.trim_zeros(*args, **kwargs)


@global_eager_registry.register("Triu")
def _triu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _triu operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    return backend_module.triu(*args, **kwargs)


@global_eager_registry.register("FillDiagonal")
def _fill_diagonal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fill_diagonal operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "fill_diagonal", None)
    if func:
        return func(*args, **kwargs)
    (x, val) = (args[0], args[1])
    import numpy as np

    x_np = np.array(x, copy=True)
    np.fill_diagonal(x_np, val, **kwargs)
    return backend_module.array(x_np) if hasattr(backend_module, "array") else x_np


@global_eager_registry.register("BandPart")
def _np_bandpart(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bandpart operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.

    Raises:
        ValueError: An exception.
    """
    func = getattr(backend_module, "bandpart", getattr(backend_module, "bandpart", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.triu(np.tril(args[0], kwargs.get("num_lower", -1)), -kwargs.get("num_upper", -1))


@global_eager_registry.register("Geometric")
def _np_geometric(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_geometric operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "geometric", getattr(backend_module, "geometric", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.geometric(*args, **kwargs)


@global_eager_registry.register("Triangular")
def _np_triangular(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_triangular operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "triangular", getattr(backend_module, "triangular", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.triangular(*args, **kwargs)


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tridiagonal operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "tridiagonal", getattr(backend_module, "tridiagonal", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.diag(args[0])


@global_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trilindices operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "trilindices", getattr(backend_module, "trilindices", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.tril_indices(*args, **kwargs)


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trilindicesfrom operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "trilindicesfrom", getattr(backend_module, "trilindicesfrom", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.tril_indices_from(*args, **kwargs)


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trimzeros operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "trimzeros", getattr(backend_module, "trimzeros", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.trim_zeros(*args, **kwargs)


@global_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_triuindices operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "triuindices", getattr(backend_module, "triuindices", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.triu_indices(*args, **kwargs)


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_triuindicesfrom operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "triuindicesfrom", getattr(backend_module, "triuindicesfrom", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.triu_indices_from(*args, **kwargs)
