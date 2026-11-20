"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_array,
    coerce_float,
    coerce_numeric,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

from .math_matrix_utils import _apply_causal_mask
from .math_reduction import _apply_softmax
from .math_special import _dispatch_special


@global_eager_registry.register("MatMul")
@global_eager_registry.register("Matmul")
def _matmul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate matrix multiplication.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Matrix product result.
    """
    mod = backend_module
    a = args[0]
    b = args[1]
    kw = dict(kwargs)
    if kw.pop("transpose_a", False):
        a = adapt_backend_module(mod).require_symbol("swapaxes")(a, -1, -2)
    if kw.pop("transpose_b", False):
        b = adapt_backend_module(mod).require_symbol("swapaxes")(b, -1, -2)
    if adapt_backend_module(backend_module).has_symbol("matmul"):
        return adapt_backend_module(mod).require_symbol("matmul")(a, b, **kw)
    return coerce_array(mod, a) @ coerce_array(mod, b)


@global_eager_registry.register("BatchMatMul")
@global_eager_registry.register("BatchMatmul")
def _batch_matmul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate batched matrix multiplication.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Batched matrix product result.
    """
    if adapt_backend_module(backend_module).has_symbol("matmul"):
        _typed_ret_1: BackendArg = adapt_backend_module(backend_module).require_symbol("matmul")(*args, **kwargs)
        return _typed_ret_1
    return coerce_array(backend_module, args[0]) @ coerce_array(backend_module, args[1])


@global_eager_registry.register("Dot")
def _dot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate dot product.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (A, B).
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Dot product result.
    """
    if adapt_backend_module(backend_module).has_symbol("dot"):
        _typed_ret_2: BackendArg = adapt_backend_module(backend_module).require_symbol("dot")(*args, **kwargs)
        return _typed_ret_2
    return coerce_array(backend_module, args[0]) @ coerce_array(backend_module, args[1])


@global_eager_registry.register("Einsum")
def _einsum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager einsum operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    eq = (
        kwargs.pop("equation", "")
        if adapt_backend_module(kwargs).has_symbol("pop") and "equation" in kwargs
        else args[0]
        if len(args) > 0 and isinstance(args[0], str)
        else ""
    )
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args
    if adapt_backend_module(backend_module).has_symbol("einsum"):
        _typed_ret_3: BackendArg = adapt_backend_module(backend_module).require_symbol("einsum")(eq, *op_args, **kwargs)
        return _typed_ret_3
    return _resolve_eager_mod(backend_module).einsum(eq, *op_args, **kwargs)


@global_eager_registry.register("ScaledDotProductAttention")
def _scaled_dot_product_attention_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    query: BackendArg,
    key: BackendArg,
    value: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Fallback eager execution for ScaledDotProductAttention.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        query: Input `query` argument (query).
        key: Input `key` argument (key).
        value: Input `value` argument (value).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    import math

    mod = backend_module
    raw_scale = kwargs.get("scale") if adapt_backend_module(kwargs).has_symbol("get") else None
    is_causal = kwargs.get("is_causal", False) if adapt_backend_module(kwargs).has_symbol("get") else False
    mask = kwargs.get("mask", None) if adapt_backend_module(kwargs).has_symbol("get") else None

    q_shape = coerce_shape(adapt_backend_module(query).require_symbol("shape"))
    k_shape = coerce_shape(adapt_backend_module(key).require_symbol("shape"))
    scale_f: float = 1.0 / math.sqrt(q_shape[-1]) if raw_scale is None else coerce_float(raw_scale)

    # key transpose
    key_t_axes = list(range(len(k_shape)))
    key_t_axes[-1], key_t_axes[-2] = key_t_axes[-2], key_t_axes[-1]

    if adapt_backend_module(backend_module).has_symbol("transpose"):
        key_t: BackendArg = adapt_backend_module(mod).require_symbol("transpose")(key, axes=key_t_axes)
    else:
        key_t = key

    scores: BackendArg = adapt_backend_module(mod).require_symbol("matmul")(query, key_t) * scale_f

    if is_causal:
        scores = _apply_causal_mask(backend_module, scores)

    if mask is not None:
        scores = coerce_numeric(scores) + coerce_numeric(mask)

    attn = _apply_softmax(backend_module, scores)

    return adapt_backend_module(mod).require_symbol("matmul")(attn, value)


@global_eager_registry.register("CholeskySolve")
def _cholesky_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cholesky solve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("cho_solve"):
        return func.cho_solve(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("cho_solve"):
        return adapt_backend_module(mod).require_symbol("cho_solve")(*args, **kwargs)

    b, c = args[0], args[1]
    return _dispatch_special(
        "CholeskySolve",
        "cho_solve",
        backend_module,
        adapt_backend_module(mod).require_symbol("asarray")(b),
        adapt_backend_module(mod).require_symbol("asarray")(c),
    )


@global_eager_registry.register("BandedTriangularSolve")
def _banded_triangular_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager banded triangular solve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("solve_banded"):
        return func.solve_banded(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("solve_banded"):
        return adapt_backend_module(mod).require_symbol("solve_banded")(*args, **kwargs)

    a, b = args[0], args[1]
    return _dispatch_special(
        "BandedTriangularSolve",
        "solve_banded",
        backend_module,
        adapt_backend_module(mod).require_symbol("asarray")(a),
        adapt_backend_module(mod).require_symbol("asarray")(b),
        **kwargs,
    )


@global_eager_registry.register("MatrixPower")
def _matrix_power(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager matrix power operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("matrix_power"):
        return func.matrix_power(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("matrix_power"):
        return adapt_backend_module(mod).require_symbol("matrix_power")(*args, **kwargs)

    x, n = args[0], args[1]
    return (
        adapt_backend_module(mod)
        .require_symbol("linalg")
        .matrix_power(adapt_backend_module(mod).require_symbol("asarray")(x), n)
    )


@global_eager_registry.register("MatrixRank")
def _matrix_rank(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager matrix rank operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("matrix_rank"):
        return func.matrix_rank(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("matrix_rank"):
        return adapt_backend_module(mod).require_symbol("matrix_rank")(*args, **kwargs)

    x = args[0]
    return (
        adapt_backend_module(mod)
        .require_symbol("linalg")
        .matrix_rank(adapt_backend_module(mod).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Solve")
def _solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager solve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("solve"):
        return func.solve(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("solve"):
        return adapt_backend_module(mod).require_symbol("solve")(*args, **kwargs)

    a, b = args[0], args[1]
    return (
        adapt_backend_module(mod)
        .require_symbol("linalg")
        .solve(
            adapt_backend_module(mod).require_symbol("asarray")(a),
            adapt_backend_module(mod).require_symbol("asarray")(b),
        )
    )


@global_eager_registry.register("Tensorsolve")
def _tensorsolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager tensorsolve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("tensorsolve"):
        return func.tensorsolve(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("tensorsolve"):
        return adapt_backend_module(mod).require_symbol("tensorsolve")(*args, **kwargs)

    a, b = args[0], args[1]
    return (
        adapt_backend_module(mod)
        .require_symbol("linalg")
        .tensorsolve(
            adapt_backend_module(mod).require_symbol("asarray")(a),
            adapt_backend_module(mod).require_symbol("asarray")(b),
            **kwargs,
        )
    )


@global_eager_registry.register("Tensorsolve")
def _np_tensorsolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorsolve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "tensorsolve", adapt_backend_module(backend_module).get_symbol("tensorsolve", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.tensorsolve(*args, **kwargs)


@global_eager_registry.register("TriangularSolve")
def _np_triangularsolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np triangularsolve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "triangularsolve", adapt_backend_module(backend_module).get_symbol("triangularsolve", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.solve(*args)


@global_eager_registry.register("TridiagonalMatmul")
def _np_tridiagonalmatmul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tridiagonalmatmul operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "tridiagonalmatmul", adapt_backend_module(backend_module).get_symbol("tridiagonalmatmul", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).matmul(args[0], args[1])


@global_eager_registry.register("TridiagonalSolve")
def _np_tridiagonalsolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tridiagonalsolve operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "tridiagonalsolve", adapt_backend_module(backend_module).get_symbol("tridiagonalsolve", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).linalg.solve(*args)


@global_eager_registry.register("Vecdot")
def _np_vecdot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vecdot operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "vecdot", adapt_backend_module(backend_module).get_symbol("vecdot", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).vdot(args[0], args[1])
