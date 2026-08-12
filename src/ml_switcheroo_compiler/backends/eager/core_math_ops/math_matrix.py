# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

from .math_matrix_utils import _apply_causal_mask
from .math_reduction import _apply_softmax


@global_eager_registry.register("Einsum")
def _einsum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _einsum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    eq = kwargs.pop("equation", "") if "equation" in kwargs else args[0] if len(args) > 0 and isinstance(args[0], str) else ""
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args
    if hasattr(backend_module, "einsum"):
        return backend_module.einsum(eq, *op_args, **kwargs)
    return None


@global_eager_registry.register("ScaledDotProductAttention")
def _scaled_dot_product_attention_eager(backend_module: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    """Fallback eager execution for ScaledDotProductAttention.

    Args:
        backend_module (object): The backend_module parameter.
        query (object): The query parameter.
        key (object): The key parameter.
        value (object): The value parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import math

    scale = kwargs.get("scale")
    is_causal = kwargs.get("is_causal", False)
    mask = kwargs.get("mask", None)

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # key transpose
    key_t_axes = list(range(len(key.shape)))
    key_t_axes[-1], key_t_axes[-2] = key_t_axes[-2], key_t_axes[-1]

    if hasattr(backend_module, "transpose"):
        key_t = backend_module.transpose(key, axes=key_t_axes)
    else:
        key_t = key

    scores = backend_module.matmul(query, key_t) * scale

    if is_causal:
        scores = _apply_causal_mask(backend_module, scores)

    if mask is not None:
        scores = scores + mask

    attn = _apply_softmax(backend_module, scores)

    return backend_module.matmul(attn, value)


@global_eager_registry.register("CholeskySolve")
def _cholesky_solve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cholesky_solve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.linalg

    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "cho_solve"):
        return func.cho_solve(*args, **kwargs)
    if hasattr(backend_module, "cho_solve"):
        return backend_module.cho_solve(*args, **kwargs)

    b, c = args[0], args[1]

    return scipy.linalg.cho_solve((backend_module.asarray(c), False), backend_module.asarray(b))


@global_eager_registry.register("BandedTriangularSolve")
def _banded_triangular_solve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _banded_triangular_solve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "solve_banded"):
        return func.solve_banded(*args, **kwargs)
    if hasattr(backend_module, "solve_banded"):
        return backend_module.solve_banded(*args, **kwargs)

    import scipy.linalg

    a, b = args[0], args[1]
    return scipy.linalg.solve_banded((1, 1), backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("Igammac")
@global_eager_registry.register("Polygamma")
@global_eager_registry.register("MatrixPower")
def _matrix_power(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _matrix_power operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "matrix_power"):
        return func.matrix_power(*args, **kwargs)
    if hasattr(backend_module, "matrix_power"):
        return backend_module.matrix_power(*args, **kwargs)

    x, n = args[0], args[1]
    return backend_module.linalg.matrix_power(backend_module.asarray(x), n)


@global_eager_registry.register("MatrixRank")
def _matrix_rank(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _matrix_rank operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "matrix_rank"):
        return func.matrix_rank(*args, **kwargs)
    if hasattr(backend_module, "matrix_rank"):
        return backend_module.matrix_rank(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.matrix_rank(backend_module.asarray(x))


@global_eager_registry.register("Solve")
def _solve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _solve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "solve"):
        return func.solve(*args, **kwargs)
    if hasattr(backend_module, "solve"):
        return backend_module.solve(*args, **kwargs)

    a, b = args[0], args[1]
    return backend_module.linalg.solve(backend_module.asarray(a), backend_module.asarray(b))


@global_eager_registry.register("Tensorsolve")
def _tensorsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _tensorsolve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "tensorsolve"):
        return func.tensorsolve(*args, **kwargs)
    if hasattr(backend_module, "tensorsolve"):
        return backend_module.tensorsolve(*args, **kwargs)

    a, b = args[0], args[1]
    return backend_module.linalg.tensorsolve(backend_module.asarray(a), backend_module.asarray(b), **kwargs)


@global_eager_registry.register("Tensorsolve")
def _np_tensorsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorsolve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorsolve", getattr(backend_module, "tensorsolve", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.tensorsolve(*args, **kwargs)


@global_eager_registry.register("TriangularSolve")
def _np_triangularsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_triangularsolve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "triangularsolve", getattr(backend_module, "triangularsolve", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.solve(*args)


@global_eager_registry.register("TridiagonalMatmul")
def _np_tridiagonalmatmul(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tridiagonalmatmul operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tridiagonalmatmul", getattr(backend_module, "tridiagonalmatmul", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.matmul(args[0], args[1])


@global_eager_registry.register("TridiagonalSolve")
def _np_tridiagonalsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tridiagonalsolve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tridiagonalsolve", getattr(backend_module, "tridiagonalsolve", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.solve(*args)


@global_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vecdot operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "vecdot", getattr(backend_module, "vecdot", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.vdot(args[0], args[1])
