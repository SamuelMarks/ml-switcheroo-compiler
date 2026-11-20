"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

from .math_nn import _global_adaptive_pool


@global_eager_registry.register("Psum")
def _psum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager psum operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(args[0])


@global_eager_registry.register("ReduceSum")
def _reduce_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate reduce sum.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        BackendArg: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if adapt_backend_module(backend_module).has_symbol("sum"):
        _typed_ret_1: BackendArg = adapt_backend_module(backend_module).require_symbol("sum")(
            args[0], axis=axis, keepdims=keepdims
        )
        return _typed_ret_1
    return args[0]


@global_eager_registry.register("ReduceMean")
def _reduce_mean(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate reduce mean.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        BackendArg: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if adapt_backend_module(backend_module).has_symbol("mean"):
        _typed_ret_2: BackendArg = adapt_backend_module(backend_module).require_symbol("mean")(
            args[0], axis=axis, keepdims=keepdims
        )
        return _typed_ret_2
    return args[0]


@global_eager_registry.register("ReduceProd")
def _reduce_prod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate reduce product.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        BackendArg: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if adapt_backend_module(backend_module).has_symbol("prod"):
        _typed_ret_3: BackendArg = adapt_backend_module(backend_module).require_symbol("prod")(
            args[0], axis=axis, keepdims=keepdims
        )
        return _typed_ret_3
    return args[0]


@global_eager_registry.register("ReduceMax")
def _reduce_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate reduce max.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        BackendArg: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if adapt_backend_module(backend_module).has_symbol("max"):
        _typed_ret_4: BackendArg = adapt_backend_module(backend_module).require_symbol("max")(
            args[0], axis=axis, keepdims=keepdims
        )
        return _typed_ret_4
    return args[0]


@global_eager_registry.register("ReduceMin")
def _reduce_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate reduce min.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        BackendArg: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if adapt_backend_module(backend_module).has_symbol("min"):
        _typed_ret_5: BackendArg = adapt_backend_module(backend_module).require_symbol("min")(
            args[0], axis=axis, keepdims=keepdims
        )
        return _typed_ret_5
    return args[0]


@global_eager_registry.register("Norm")
def _norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Frobenius or Euclidean norm.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (ord, axis, keepdims).

    Returns:
        BackendArg: Norm result.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("linalg") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("linalg")
    ).has_symbol("norm"):
        _typed_ret_6: BackendArg = adapt_backend_module(mod).require_symbol("linalg").norm(*args, **kwargs)
        return _typed_ret_6
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    return adapt_backend_module(mod).require_symbol("sqrt")(
        adapt_backend_module(mod).require_symbol("sum")(coerce_numeric(args[0]) ** 2)
    )


@global_eager_registry.register("Numel")
def _numel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate numel (total number of elements).

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: Total count of elements.
    """
    del kwargs
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    x = args[0]
    x_adp = adapt_backend_module(x)
    if x_adp.has_symbol("size"):
        return coerce_float(x_adp.require_symbol("size"))
    if x_adp.has_symbol("numel"):
        return coerce_float(x_adp.require_symbol("numel")())
    return coerce_float(adapt_backend_module(_resolve_eager_mod(backend_module).asarray(x)).require_symbol("size"))


@global_eager_registry.register("Pmean")
def _pmean(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager pmean operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(args[0])


@global_eager_registry.register("BroadcastReduce")
def _broadcast_reduce(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Reduce cotangent tensor along dimensions that were broadcast relative to target.

    Args:
        backend_module: The backend module.
        *args: Positional args (cotangent, target).
        **kwargs: Keyword args.

    Returns:
        BackendArg: Resulting reduced tensor.
    """
    del kwargs
    if len(args) < 2:
        return args[0] if args else None
    mod = backend_module
    cot = args[0]
    tgt = args[1]
    cot_shape = coerce_shape(adapt_backend_module(cot).get_symbol("shape", ()))
    tgt_shape = coerce_shape(adapt_backend_module(tgt).get_symbol("shape", ()))
    if cot_shape == tgt_shape:
        return cot
    cot_ndim = len(cot_shape)
    tgt_ndim = len(tgt_shape)
    res = cot
    if cot_ndim > tgt_ndim:
        axes_to_sum = tuple(range(cot_ndim - tgt_ndim))
        res = adapt_backend_module(mod).require_symbol("sum")(res, axis=axes_to_sum)
    cot_shape_now = coerce_shape(adapt_backend_module(res).get_symbol("shape", ()))
    for i, (cd, td) in enumerate(zip(cot_shape_now, tgt_shape)):
        if td == 1 and cd > 1:
            res = adapt_backend_module(mod).require_symbol("sum")(res, axis=i, keepdims=True)
    return res


@global_eager_registry.register("BroadcastLike")
def _broadcast_like(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Broadcast source tensor to match the shape of target tensor.

    Args:
        backend_module: The backend module.
        *args: Positional args (source, target).
        **kwargs: Keyword args.

    Returns:
        BackendArg: Resulting broadcasted tensor.
    """
    del kwargs
    if len(args) < 2:
        return args[0] if args else None
    src = args[0]
    tgt = args[1]
    tgt_shape = adapt_backend_module(tgt).get_symbol("shape", ())
    if adapt_backend_module(src).get_symbol("shape", ()) == tgt_shape:
        return src
    if adapt_backend_module(backend_module).has_symbol("broadcast_to"):
        _typed_ret_7: BackendArg = adapt_backend_module(backend_module).require_symbol("broadcast_to")(src, tgt_shape)
        return _typed_ret_7
    return src


@global_eager_registry.register("SegmentSum")
def _segment_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager segment sum operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if len(args) < 2:
        return adapt_backend_module(mod).require_symbol("asarray")(args[0]) if args else None
    data: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("asarray")(args[0])
    segment_ids: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("asarray")(args[1])
    num_segments = (
        kwargs.get(
            "num_segments",
            args[2] if len(args) > 2 else adapt_backend_module(mod).require_symbol("max")(segment_ids) + 1,
        )
        if adapt_backend_module(kwargs).has_symbol("get")
        else (args[2] if len(args) > 2 else adapt_backend_module(mod).require_symbol("max")(segment_ids) + 1)
    )

    out = adapt_backend_module(mod).require_symbol("zeros")((num_segments,) + data.shape[1:], dtype=data.dtype)
    adapt_backend_module(mod).require_symbol("add").at(out, segment_ids, data)
    return adapt_backend_module(mod).require_symbol("asarray")(out)


def _apply_softmax(backend_module: BackendModuleProtocol | types.ModuleType | None, scores: BackendArg) -> BackendArg:
    """Apply softmax to attention scores.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        scores: Input `scores` argument (scores).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("softmax"):
        return adapt_backend_module(mod).require_symbol("softmax")(scores, axis=-1)
    if adapt_backend_module(backend_module).has_symbol("nn") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("nn")
    ).has_symbol("softmax"):
        _typed_ret_8: BackendArg = adapt_backend_module(mod).require_symbol("nn").softmax(scores, axis=-1)
        return _typed_ret_8
    if (
        adapt_backend_module(backend_module).has_symbol("exp")
        and adapt_backend_module(backend_module).has_symbol("sum")
        and adapt_backend_module(backend_module).has_symbol("max")
    ):
        exps: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("exp")(
            scores - adapt_backend_module(mod).require_symbol("max")(scores, axis=-1, keepdims=True)
        )
        return exps / adapt_backend_module(mod).require_symbol("sum")(exps, axis=-1, keepdims=True)
    return scores


@global_eager_registry.register("Fmax")
def _fmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fmax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fmax", adapt_backend_module(backend_module).get_symbol("maximum", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmin")
def _fmin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fmin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fmin", adapt_backend_module(backend_module).get_symbol("minimum", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AdaptiveMaxPool2D")
def _adaptive_max_pool2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive max pool2d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D")
def _adaptive_max_pool3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive max pool3d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _adaptive_max_pool3d_indices(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive max pool3d indices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    res = _global_adaptive_pool(backend_module, operand, output_size, **kwargs)
    return (res, res)


@global_eager_registry.register("AdaptiveLogSoftmaxWithLoss")
def _adaptive_log_softmax_with_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    input: BackendArg,
    target: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive log softmax with loss operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        input: Input `input` argument (input).
        target: Input `target` argument (target).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    loss = (
        adapt_backend_module(backend_module).require_symbol("zeros")(
            (), dtype=adapt_backend_module(target).get_symbol("dtype", None)
        )
        if adapt_backend_module(backend_module).has_symbol("zeros")
        else 0.0
    )
    return (target, loss)


@global_eager_registry.register("HouseholderProduct")
def _householder_product(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager householder product operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("householder_product"):
        return func.householder_product(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("householder_product"):
        return adapt_backend_module(mod).require_symbol("householder_product")(*args, **kwargs)

    v, tau = (
        adapt_backend_module(mod).require_symbol("asarray")(args[0]),
        adapt_backend_module(mod).require_symbol("asarray")(args[1]),
    )
    m, n = v.shape[-2:]
    k = tau.shape[-1]

    batch_shape = v.shape[:-2]
    identity = (
        adapt_backend_module(mod)
        .require_symbol("broadcast_to")(
            adapt_backend_module(mod).require_symbol("eye")(m, dtype=v.dtype), batch_shape + (m, m)
        )
        .copy()
    )
    q = identity.copy()

    for i in range(k):
        v_i = v[..., :, i].copy()
        v_i[..., :i] = 0
        v_i[..., i] = 1

        newaxis = adapt_backend_module(mod).get_symbol("newaxis", None)
        v_i_expanded = v_i[..., newaxis]
        v_i_h = adapt_backend_module(backend_module).require_symbol("conjugate")(v_i_expanded.swapaxes(-1, -2))

        tau_i = tau[
            ...,
            i,
            newaxis,
            newaxis,
        ]

        h_i = identity - tau_i * (v_i_expanded @ v_i_h)
        q = q @ h_i

    return q[..., :n]


@global_eager_registry.register("Cummax")
def _cummax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cummax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("maximum", None)
    if func and adapt_backend_module(func).has_symbol("accumulate"):
        return func.accumulate(*args, **kwargs)

    return (
        adapt_backend_module(mod)
        .require_symbol("maximum")
        .accumulate(adapt_backend_module(mod).require_symbol("asarray")(args[0]), **kwargs)
    )


@global_eager_registry.register("Cummin")
def _cummin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cummin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("minimum", None)
    if func and adapt_backend_module(func).has_symbol("accumulate"):
        return func.accumulate(*args, **kwargs)

    return (
        adapt_backend_module(mod)
        .require_symbol("minimum")
        .accumulate(adapt_backend_module(mod).require_symbol("asarray")(args[0]), **kwargs)
    )


@global_eager_registry.register("Cumlogsumexp")
def _cumlogsumexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cumlogsumexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("cumlogsumexp", None)
    if func:
        return func(*args, **kwargs)

    x = adapt_backend_module(mod).require_symbol("asarray")(args[0])
    axis = kwargs.get("axis", 0)

    return (
        adapt_backend_module(mod)
        .require_symbol("ufunc")
        .accumulate(adapt_backend_module(mod).require_symbol("logaddexp"), x, axis=axis)
    )


@global_eager_registry.register("CumulativeLogsumexp")
def _cumulative_logsumexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager cumulative logsumexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _cumlogsumexp(backend_module, *args, **kwargs)


@global_eager_registry.register("PsumScatter")
def _psumscatter(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager psumscatter operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    if adapt_backend_module(backend_module).has_symbol("lax") and adapt_backend_module(
        adapt_backend_module(mod).require_symbol("lax")
    ).has_symbol("psum_scatter"):
        _typed_ret_9: BackendArg = adapt_backend_module(mod).require_symbol("lax").psum_scatter(*args, **kwargs)
        return _typed_ret_9
    return args[0] if args else None


@global_eager_registry.register("Fmax")
def _np_fmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fmax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fmax", adapt_backend_module(backend_module).get_symbol("fmax", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).fmax(args[0], args[1])


@global_eager_registry.register("ScatterMax")
def _np_scattermax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scattermax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "scattermax", adapt_backend_module(backend_module).get_symbol("scattermax", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("ScatterMin")
def _np_scattermin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scattermin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "scattermin", adapt_backend_module(backend_module).get_symbol("scattermin", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("WeibullMin")
def _np_weibullmin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np weibullmin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "weibullmin", adapt_backend_module(backend_module).get_symbol("weibullmin", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).random.weibull(*args, **kwargs)


@global_eager_registry.register("WindowHamming")
def _np_windowhamming(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np windowhamming operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "windowhamming", adapt_backend_module(backend_module).get_symbol("windowhamming", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).hamming(*args, **kwargs)
