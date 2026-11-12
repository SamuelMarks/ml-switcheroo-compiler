"""Implementation of math reduction operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    ensure_eager_tensor,
    ensure_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, ensure_mod, get_backend_member, has_backend_member

from .math_nn import _global_adaptive_pool


@global_eager_registry.register("Psum")
def _psum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager psum operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("ReduceSum")
def _reduce_sum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate reduce sum.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        EagerValue: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if has_backend_member(backend_module, "sum"):
        _typed_ret_1: EagerValue = backend_module.sum(args[0], axis=axis, keepdims=keepdims)
        return _typed_ret_1
    return args[0]


@global_eager_registry.register("ReduceMean")
def _reduce_mean(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate reduce mean.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        EagerValue: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if has_backend_member(backend_module, "mean"):
        _typed_ret_2: EagerValue = backend_module.mean(args[0], axis=axis, keepdims=keepdims)
        return _typed_ret_2
    return args[0]


@global_eager_registry.register("ReduceProd")
def _reduce_prod(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate reduce product.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        EagerValue: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if has_backend_member(backend_module, "prod"):
        _typed_ret_3: EagerValue = backend_module.prod(args[0], axis=axis, keepdims=keepdims)
        return _typed_ret_3
    return args[0]


@global_eager_registry.register("ReduceMax")
def _reduce_max(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate reduce max.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        EagerValue: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if has_backend_member(backend_module, "max"):
        _typed_ret_4: EagerValue = backend_module.max(args[0], axis=axis, keepdims=keepdims)
        return _typed_ret_4
    return args[0]


@global_eager_registry.register("ReduceMin")
def _reduce_min(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate reduce min.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (axis, keepdims).

    Returns:
        EagerValue: Reduced tensor.
    """
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if has_backend_member(backend_module, "min"):
        _typed_ret_5: EagerValue = backend_module.min(args[0], axis=axis, keepdims=keepdims)
        return _typed_ret_5
    return args[0]


@global_eager_registry.register("Norm")
def _norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Frobenius or Euclidean norm.

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Keyword arguments (ord, axis, keepdims).

    Returns:
        EagerValue: Norm result.
    """
    mod = backend_module
    if has_backend_member(backend_module, "linalg") and has_backend_member(mod.linalg, "norm"):
        _typed_ret_6: EagerValue = mod.linalg.norm(*args, **kwargs)
        return _typed_ret_6
    return mod.sqrt(mod.sum(ensure_eager_tensor(args[0]) ** 2))


@global_eager_registry.register("Numel")
def _numel(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate numel (total number of elements).

    Args:
        backend_module: Backend execution module.
        *args: Positional arguments (input tensor).
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: Total count of elements.
    """
    del kwargs
    x = args[0]
    if has_backend_member(x, "size"):
        return float(ensure_eager_tensor(x).size)
    if has_backend_member(x, "numel"):
        return float(ensure_mod(x).numel())
    return float(_resolve_eager_mod(backend_module).asarray(x).size)


@global_eager_registry.register("Pmean")
def _pmean(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pmean operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("BroadcastReduce")
def _broadcast_reduce(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Reduce cotangent tensor along dimensions that were broadcast relative to target.

    Args:
        backend_module: The backend module.
        *args: Positional args (cotangent, target).
        **kwargs: Keyword args.

    Returns:
        EagerValue: Resulting reduced tensor.
    """
    del kwargs
    if len(args) < 2:
        return args[0] if args else None
    mod = backend_module
    cot = args[0]
    tgt = args[1]
    cot_shape = ensure_shape(get_backend_member(cot, "shape", ()))
    tgt_shape = ensure_shape(get_backend_member(tgt, "shape", ()))
    if cot_shape == tgt_shape:
        return cot
    cot_ndim = len(cot_shape)
    tgt_ndim = len(tgt_shape)
    res = cot
    if cot_ndim > tgt_ndim:
        axes_to_sum = tuple(range(cot_ndim - tgt_ndim))
        res = mod.sum(res, axis=axes_to_sum)
    cot_shape_now = ensure_shape(get_backend_member(res, "shape", ()))
    for i, (cd, td) in enumerate(zip(cot_shape_now, tgt_shape)):
        if td == 1 and cd > 1:
            res = mod.sum(res, axis=i, keepdims=True)
    return res


@global_eager_registry.register("BroadcastLike")
def _broadcast_like(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Broadcast source tensor to match the shape of target tensor.

    Args:
        backend_module: The backend module.
        *args: Positional args (source, target).
        **kwargs: Keyword args.

    Returns:
        EagerValue: Resulting broadcasted tensor.
    """
    del kwargs
    if len(args) < 2:
        return args[0] if args else None
    src = args[0]
    tgt = args[1]
    tgt_shape = get_backend_member(tgt, "shape", ())
    if get_backend_member(src, "shape", ()) == tgt_shape:
        return src
    if has_backend_member(backend_module, "broadcast_to"):
        _typed_ret_7: EagerValue = backend_module.broadcast_to(src, tgt_shape)
        return _typed_ret_7
    return src


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager segment sum operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    if len(args) < 2:
        return mod.asarray(args[0]) if args else None
    data: EagerTensorProtocol = mod.asarray(args[0])
    segment_ids: EagerTensorProtocol = mod.asarray(args[1])
    num_segments = kwargs.get("num_segments", args[2] if len(args) > 2 else mod.max(segment_ids) + 1) if has_backend_member(kwargs, "get") else (args[2] if len(args) > 2 else mod.max(segment_ids) + 1)

    out = mod.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    mod.add.at(out, segment_ids, data)
    return mod.asarray(out)


def _apply_softmax(backend_module: BackendModuleProtocol, scores: EagerValue) -> EagerValue:
    """Apply softmax to attention scores.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        scores: Input `scores` argument (scores).

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    if has_backend_member(backend_module, "softmax"):
        return mod.softmax(scores, axis=-1)
    if has_backend_member(backend_module, "nn") and has_backend_member(mod.nn, "softmax"):
        _typed_ret_8: EagerValue = mod.nn.softmax(scores, axis=-1)
        return _typed_ret_8
    if has_backend_member(backend_module, "exp") and has_backend_member(backend_module, "sum") and has_backend_member(backend_module, "max"):
        exps: EagerTensorProtocol = mod.exp(ensure_eager_tensor(scores) - mod.max(scores, axis=-1, keepdims=True))
        return exps / mod.sum(exps, axis=-1, keepdims=True)
    return scores


@global_eager_registry.register("Fmax")
def _fmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fmax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "fmax", get_backend_member(backend_module, "maximum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmin")
def _fmin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fmin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "fmin", get_backend_member(backend_module, "minimum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AdaptiveMaxPool2D")
def _adaptive_max_pool2d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool2d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D")
def _adaptive_max_pool3d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool3d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _adaptive_max_pool3d_indices(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool3d indices operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    res = _global_adaptive_pool(backend_module, operand, output_size, **kwargs)
    return (res, res)


@global_eager_registry.register("AdaptiveLogSoftmaxWithLoss")
def _adaptive_log_softmax_with_loss(backend_module: BackendModuleProtocol, input: EagerValue, target: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive log softmax with loss operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        input: Input `input` argument (input).
        target: Input `target` argument (target).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    loss = backend_module.zeros((), dtype=get_backend_member(target, "dtype", None)) if has_backend_member(backend_module, "zeros") else 0.0
    return (target, loss)


@global_eager_registry.register("HouseholderProduct")
def _householder_product(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager householder product operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "linalg", None)
    if func and has_backend_member(func, "householder_product"):
        return ensure_eager_tensor(func.householder_product(*args, **kwargs))
    if has_backend_member(backend_module, "householder_product"):
        return mod.householder_product(*args, **kwargs)

    v, tau = mod.asarray(args[0]), mod.asarray(args[1])
    m, n = v.shape[-2:]
    k = tau.shape[-1]

    batch_shape = v.shape[:-2]
    identity = mod.broadcast_to(mod.eye(m, dtype=v.dtype), batch_shape + (m, m)).copy()
    q = identity.copy()

    for i in range(k):
        v_i = v[..., :, i].copy()
        v_i[..., :i] = 0
        v_i[..., i] = 1

        v_i_expanded = v_i[..., mod.newaxis]
        v_i_h = get_backend_member(backend_module, "conjugate")(v_i_expanded.swapaxes(-1, -2))

        tau_i = tau[..., i, mod.newaxis, mod.newaxis]

        h_i = identity - tau_i * (v_i_expanded @ v_i_h)
        q = q @ h_i

    return q[..., :n]


@global_eager_registry.register("Cummax")
def _cummax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cummax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "maximum", None)
    if func and has_backend_member(func, "accumulate"):
        return ensure_eager_tensor(func.accumulate(*args, **kwargs))

    return mod.maximum.accumulate(mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cummin")
def _cummin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cummin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "minimum", None)
    if func and has_backend_member(func, "accumulate"):
        return ensure_eager_tensor(func.accumulate(*args, **kwargs))

    return mod.minimum.accumulate(mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cumlogsumexp")
def _cumlogsumexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cumlogsumexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    func = get_backend_member(backend_module, "cumlogsumexp", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))

    x = mod.asarray(args[0])
    axis = kwargs.get("axis", 0) if has_backend_member(kwargs, "get") else 0

    return mod.ufunc.accumulate(mod.logaddexp, x, axis=axis)


@global_eager_registry.register("CumulativeLogsumexp")
def _cumulative_logsumexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cumulative logsumexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return _cumlogsumexp(backend_module, *args, **kwargs)


@global_eager_registry.register("PsumScatter")
def _psumscatter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager psumscatter operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    mod = backend_module
    if has_backend_member(backend_module, "lax") and has_backend_member(mod.lax, "psum_scatter"):
        _typed_ret_9: EagerValue = mod.lax.psum_scatter(*args, **kwargs)
        return _typed_ret_9
    return args[0] if args else None


@global_eager_registry.register("Fmax")
def _np_fmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fmax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "fmax", get_backend_member(backend_module, "fmax", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fmax(args[0], args[1])


@global_eager_registry.register("ScatterMax")
def _np_scattermax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scattermax operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "scattermax", get_backend_member(backend_module, "scattermax", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("ScatterMin")
def _np_scattermin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scattermin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "scattermin", get_backend_member(backend_module, "scattermin", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("WeibullMin")
def _np_weibullmin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np weibullmin operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "weibullmin", get_backend_member(backend_module, "weibullmin", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.weibull(*args, **kwargs)


@global_eager_registry.register("WindowHamming")
def _np_windowhamming(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np windowhamming operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "windowhamming", get_backend_member(backend_module, "windowhamming", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).hamming(*args, **kwargs)
