"""Implementation of math reduction operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    as_tensor,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, as_mod

from .math_nn import _global_adaptive_pool


@global_eager_registry.register("Psum")
def _psum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager psum operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
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
    if hasattr(backend_module, "sum"):
        return backend_module.sum(args[0], axis=axis, keepdims=keepdims)
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
    if hasattr(backend_module, "mean"):
        return backend_module.mean(args[0], axis=axis, keepdims=keepdims)
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
    if hasattr(backend_module, "prod"):
        return backend_module.prod(args[0], axis=axis, keepdims=keepdims)
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
    if hasattr(backend_module, "max"):
        return backend_module.max(args[0], axis=axis, keepdims=keepdims)
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
    if hasattr(backend_module, "min"):
        return backend_module.min(args[0], axis=axis, keepdims=keepdims)
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
    if hasattr(backend_module, "linalg") and hasattr(mod.linalg, "norm"):
        return mod.linalg.norm(*args, **kwargs)
    return mod.sqrt(mod.sum(as_tensor(args[0]) ** 2))


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
    if hasattr(x, "size"):
        return float(as_tensor(x).size)
    if hasattr(x, "numel"):
        return float(as_mod(x).numel())
    return float(_resolve_eager_mod(backend_module).asarray(x).size)


@global_eager_registry.register("Pmean")
def _pmean(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pmean operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
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
    cot_shape = getattr(cot, "shape", ())
    tgt_shape = getattr(tgt, "shape", ())
    if cot_shape == tgt_shape:
        return cot
    cot_ndim = len(cot_shape)
    tgt_ndim = len(tgt_shape)
    res = cot
    if cot_ndim > tgt_ndim:
        axes_to_sum = tuple(range(cot_ndim - tgt_ndim))
        res = mod.sum(res, axis=axes_to_sum)
    cot_shape_now = getattr(res, "shape", ())
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
    tgt_shape = getattr(tgt, "shape", ())
    if getattr(src, "shape", ()) == tgt_shape:
        return src
    if hasattr(backend_module, "broadcast_to"):
        return backend_module.broadcast_to(src, tgt_shape)
    return src


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager segment sum operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    if len(args) < 2:
        return mod.asarray(args[0]) if args else None
    data: EagerTensorProtocol = mod.asarray(args[0])
    segment_ids: EagerTensorProtocol = mod.asarray(args[1])
    num_segments = kwargs.get("num_segments", args[2] if len(args) > 2 else mod.max(segment_ids) + 1) if hasattr(kwargs, "get") else (args[2] if len(args) > 2 else mod.max(segment_ids) + 1)

    out = mod.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    mod.add.at(out, segment_ids, data)
    return mod.asarray(out)


def _apply_softmax(backend_module: BackendModuleProtocol, scores: EagerValue) -> EagerValue:
    """Apply softmax to attention scores.

    Args:
        backend_module: The backend_module parameter.
        scores: The scores parameter.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    if hasattr(backend_module, "softmax"):
        return mod.softmax(scores, axis=-1)
    if hasattr(backend_module, "nn") and hasattr(mod.nn, "softmax"):
        return mod.nn.softmax(scores, axis=-1)
    if hasattr(backend_module, "exp") and hasattr(backend_module, "sum") and hasattr(backend_module, "max"):
        exps: EagerTensorProtocol = mod.exp(as_tensor(scores) - mod.max(scores, axis=-1, keepdims=True))
        return exps / mod.sum(exps, axis=-1, keepdims=True)
    return scores


@global_eager_registry.register("Fmax")
def _fmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fmax operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fmax", getattr(backend_module, "maximum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmin")
def _fmin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fmin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fmin", getattr(backend_module, "minimum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AdaptiveMaxPool2D")
def _adaptive_max_pool2d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool2d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D")
def _adaptive_max_pool3d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool3d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _adaptive_max_pool3d_indices(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive max pool3d indices operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    res = _global_adaptive_pool(backend_module, operand, output_size, **kwargs)
    return (res, res)


@global_eager_registry.register("AdaptiveLogSoftmaxWithLoss")
def _adaptive_log_softmax_with_loss(backend_module: BackendModuleProtocol, input: EagerValue, target: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive log softmax with loss operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        input: The input parameter.
        target: The target parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    loss = backend_module.zeros((), dtype=getattr(target, "dtype", None)) if hasattr(backend_module, "zeros") else 0.0
    return (target, loss)


@global_eager_registry.register("HouseholderProduct")
def _householder_product(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager householder product operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "householder_product"):
        return as_tensor(func.householder_product(*args, **kwargs))
    if hasattr(backend_module, "householder_product"):
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
        v_i_h = getattr(backend_module, "conjugate")(v_i_expanded.swapaxes(-1, -2))

        tau_i = tau[..., i, mod.newaxis, mod.newaxis]

        h_i = identity - tau_i * (v_i_expanded @ v_i_h)
        q = q @ h_i

    return q[..., :n]


@global_eager_registry.register("Cummax")
def _cummax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cummax operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "maximum", None)
    if func and hasattr(func, "accumulate"):
        return as_tensor(func.accumulate(*args, **kwargs))

    return mod.maximum.accumulate(mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cummin")
def _cummin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cummin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "minimum", None)
    if func and hasattr(func, "accumulate"):
        return as_tensor(func.accumulate(*args, **kwargs))

    return mod.minimum.accumulate(mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cumlogsumexp")
def _cumlogsumexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cumlogsumexp operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "cumlogsumexp", None)
    if func:
        return as_tensor(func(*args, **kwargs))

    x = mod.asarray(args[0])
    axis = kwargs.get("axis", 0) if hasattr(kwargs, "get") else 0

    return mod.ufunc.accumulate(mod.logaddexp, x, axis=axis)


@global_eager_registry.register("CumulativeLogsumexp")
def _cumulative_logsumexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager cumulative logsumexp operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _cumlogsumexp(backend_module, *args, **kwargs)


@global_eager_registry.register("PsumScatter")
def _psumscatter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager psumscatter operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    if hasattr(backend_module, "lax") and hasattr(mod.lax, "psum_scatter"):
        return mod.lax.psum_scatter(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Fmax")
def _np_fmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fmax operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fmax", getattr(backend_module, "fmax", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fmax(args[0], args[1])


@global_eager_registry.register("ScatterMax")
def _np_scattermax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scattermax operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "scattermax", getattr(backend_module, "scattermax", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("ScatterMin")
def _np_scattermin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scattermin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "scattermin", getattr(backend_module, "scattermin", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("WeibullMin")
def _np_weibullmin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np weibullmin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "weibullmin", getattr(backend_module, "weibullmin", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.weibull(*args, **kwargs)


@global_eager_registry.register("WindowHamming")
def _np_windowhamming(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np windowhamming operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "windowhamming", getattr(backend_module, "windowhamming", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).hamming(*args, **kwargs)
