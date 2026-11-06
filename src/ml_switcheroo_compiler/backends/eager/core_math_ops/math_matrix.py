"""Implementation of math matrix operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    ensure_eager_tensor,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member

from .math_matrix_utils import _apply_causal_mask
from .math_reduction import _apply_softmax
from .math_special import _dispatch_special


@global_eager_registry.register("MatMul")
@global_eager_registry.register("Matmul")
def _matmul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix multiplication.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix product result.
    """
    mod = backend_module
    a = ensure_eager_tensor(args[0])
    b = ensure_eager_tensor(args[1])
    kw = dict(kwargs)
    if kw.pop("transpose_a", False):
        a = mod.swapaxes(a, -1, -2)
    if kw.pop("transpose_b", False):
        b = mod.swapaxes(b, -1, -2)
    if has_backend_member(backend_module, "matmul"):
        return mod.matmul(a, b, **kw)
    return a @ b


@global_eager_registry.register("BatchMatMul")
@global_eager_registry.register("BatchMatmul")
def _batch_matmul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate batched matrix multiplication.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Batched matrix product result.
    """
    if has_backend_member(backend_module, "matmul"):
        _typed_ret_1: EagerValue = backend_module.matmul(*args, **kwargs)
        return _typed_ret_1
    return ensure_eager_tensor(args[0]) @ ensure_eager_tensor(args[1])


@global_eager_registry.register("Dot")
def _dot(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate dot product.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Dot product result.
    """
    if has_backend_member(backend_module, "dot"):
        _typed_ret_2: EagerValue = backend_module.dot(*args, **kwargs)
        return _typed_ret_2
    return ensure_eager_tensor(args[0]) @ ensure_eager_tensor(args[1])


@global_eager_registry.register("Einsum")
def _einsum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager einsum operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    eq = kwargs.pop("equation", "") if has_backend_member(kwargs, "pop") and "equation" in kwargs else args[0] if len(args) > 0 and isinstance(args[0], str) else ""
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args
    if has_backend_member(backend_module, "einsum"):
        _typed_ret_3: EagerValue = backend_module.einsum(eq, *op_args, **kwargs)
        return _typed_ret_3
    return _resolve_eager_mod(backend_module).einsum(eq, *op_args, **kwargs)


@global_eager_registry.register("ScaledDotProductAttention")
def _scaled_dot_product_attention_eager(backend_module: BackendModuleProtocol, query: EagerValue, key: EagerValue, value: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Fallback eager execution for ScaledDotProductAttention.

    Args:
        backend_module: The backend_module parameter.
        query: The query parameter.
        key: The key parameter.
        value: The value parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    import math

    mod = backend_module
    scale = kwargs.get("scale") if has_backend_member(kwargs, "get") else None
    is_causal = kwargs.get("is_causal", False) if has_backend_member(kwargs, "get") else False
    mask = kwargs.get("mask", None) if has_backend_member(kwargs, "get") else None

    if scale is None:
        scale = 1.0 / math.sqrt(ensure_eager_tensor(query).shape[-1])

    # key transpose
    key_t_axes = list(range(len(ensure_eager_tensor(key).shape)))
    key_t_axes[-1], key_t_axes[-2] = key_t_axes[-2], key_t_axes[-1]

    if has_backend_member(backend_module, "transpose"):
        key_t: EagerTensorProtocol = mod.transpose(key, axes=key_t_axes)
    else:
        key_t = ensure_eager_tensor(key)

    scores: EagerTensorProtocol = mod.matmul(query, key_t) * scale

    if is_causal:
        scores = ensure_eager_tensor(_apply_causal_mask(backend_module, scores))

    if mask is not None:
        scores = scores + ensure_eager_tensor(mask)

    attn = _apply_softmax(backend_module, scores)

    return mod.matmul(attn, value)


@global_eager_registry.register("CholeskySolve")
def _cholesky_solve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cholesky solve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "cho_solve"):
        return ensure_eager_tensor(func.cho_solve(*args, **kwargs))
    if has_backend_member(backend_module, "cho_solve"):
        return mod.cho_solve(*args, **kwargs)

    b, c = args[0], args[1]
    return _dispatch_special("CholeskySolve", "cho_solve", backend_module, mod.asarray(b), mod.asarray(c))


@global_eager_registry.register("BandedTriangularSolve")
def _banded_triangular_solve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager banded triangular solve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "solve_banded"):
        return ensure_eager_tensor(func.solve_banded(*args, **kwargs))
    if has_backend_member(backend_module, "solve_banded"):
        return mod.solve_banded(*args, **kwargs)

    a, b = args[0], args[1]
    return _dispatch_special("BandedTriangularSolve", "solve_banded", backend_module, mod.asarray(a), mod.asarray(b), **kwargs)


@global_eager_registry.register("MatrixPower")
def _matrix_power(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager matrix power operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "matrix_power"):
        return ensure_eager_tensor(func.matrix_power(*args, **kwargs))
    if has_backend_member(backend_module, "matrix_power"):
        return mod.matrix_power(*args, **kwargs)

    x, n = args[0], args[1]
    return mod.linalg.matrix_power(mod.asarray(x), n)


@global_eager_registry.register("MatrixRank")
def _matrix_rank(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager matrix rank operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "matrix_rank"):
        return ensure_eager_tensor(func.matrix_rank(*args, **kwargs))
    if has_backend_member(backend_module, "matrix_rank"):
        return mod.matrix_rank(*args, **kwargs)

    x = args[0]
    return mod.linalg.matrix_rank(mod.asarray(x))


@global_eager_registry.register("Solve")
def _solve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager solve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "solve"):
        return ensure_eager_tensor(func.solve(*args, **kwargs))
    if has_backend_member(backend_module, "solve"):
        return mod.solve(*args, **kwargs)

    a, b = args[0], args[1]
    return mod.linalg.solve(mod.asarray(a), mod.asarray(b))


@global_eager_registry.register("Tensorsolve")
def _tensorsolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tensorsolve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "tensorsolve"):
        return ensure_eager_tensor(func.tensorsolve(*args, **kwargs))
    if has_backend_member(backend_module, "tensorsolve"):
        return mod.tensorsolve(*args, **kwargs)

    a, b = args[0], args[1]
    return mod.linalg.tensorsolve(mod.asarray(a), mod.asarray(b), **kwargs)


@global_eager_registry.register("Tensorsolve")
def _np_tensorsolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorsolve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "tensorsolve", get_backend_member(backend_module, "tensorsolve", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.tensorsolve(*args, **kwargs)


@global_eager_registry.register("TriangularSolve")
def _np_triangularsolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np triangularsolve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "triangularsolve", get_backend_member(backend_module, "triangularsolve", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.solve(*args)


@global_eager_registry.register("TridiagonalMatmul")
def _np_tridiagonalmatmul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonalmatmul operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "tridiagonalmatmul", get_backend_member(backend_module, "tridiagonalmatmul", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).matmul(args[0], args[1])


@global_eager_registry.register("TridiagonalSolve")
def _np_tridiagonalsolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonalsolve operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "tridiagonalsolve", get_backend_member(backend_module, "tridiagonalsolve", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).linalg.solve(*args)


@global_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vecdot operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "vecdot", get_backend_member(backend_module, "vecdot", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).vdot(args[0], args[1])
