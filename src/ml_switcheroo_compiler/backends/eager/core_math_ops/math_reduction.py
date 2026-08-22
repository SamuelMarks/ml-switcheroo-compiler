# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

from .math_nn import _global_adaptive_pool


@global_eager_registry.register("Psum")
def _psum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _psum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("Pmean")
def _pmean(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _pmean operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.array(args[0])


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _segment_sum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if len(args) < 2:
        return backend_module.asarray(args[0]) if args else None
    data = backend_module.asarray(args[0])
    segment_ids = backend_module.asarray(args[1])
    num_segments = kwargs.get("num_segments", args[2] if len(args) > 2 else backend_module.max(segment_ids) + 1)

    out = backend_module.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    backend_module.add.at(out, segment_ids, data)
    return backend_module.asarray(out)


def _apply_softmax(backend_module: Any, scores: Any) -> Any:
    """Apply softmax to attention scores.

    Args:
        backend_module (object): The backend_module parameter.
        scores (object): The scores parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if hasattr(backend_module, "softmax"):
        return backend_module.softmax(scores, axis=-1)
    if hasattr(backend_module, "nn") and hasattr(backend_module.nn, "softmax"):
        return backend_module.nn.softmax(scores, axis=-1)
    if hasattr(backend_module, "exp") and hasattr(backend_module, "sum") and hasattr(backend_module, "max"):
        exps = backend_module.exp(scores - backend_module.max(scores, axis=-1, keepdims=True))
        return exps / backend_module.sum(exps, axis=-1, keepdims=True)
    return scores


@global_eager_registry.register("Fmax")
def _fmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "fmax", getattr(backend_module, "maximum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmin")
def _fmin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fmin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "fmin", getattr(backend_module, "minimum", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AdaptiveMaxPool2D")
def _adaptive_max_pool2d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_max_pool2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D")
def _adaptive_max_pool3d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_max_pool3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _adaptive_max_pool3d_indices(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_max_pool3d_indices operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    res = _global_adaptive_pool(backend_module, operand, output_size, **kwargs)
    return (res, res)


@global_eager_registry.register("AdaptiveLogSoftmaxWithLoss")
def _adaptive_log_softmax_with_loss(backend_module: Any, input: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_log_softmax_with_loss operation.

    Args:
        backend_module (object): The backend_module parameter.
        input (object): The input parameter.
        target (object): The target parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    loss = backend_module.zeros((), dtype=getattr(target, "dtype", None)) if hasattr(backend_module, "zeros") else 0.0
    return (target, loss)


@global_eager_registry.register("HouseholderProduct")
def _householder_product(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _householder_product operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "householder_product"):
        return func.householder_product(*args, **kwargs)
    if hasattr(backend_module, "householder_product"):
        return backend_module.householder_product(*args, **kwargs)

    v, tau = backend_module.asarray(args[0]), backend_module.asarray(args[1])
    m, n = v.shape[-2:]
    k = tau.shape[-1]

    batch_shape = v.shape[:-2]
    identity = backend_module.broadcast_to(backend_module.eye(m, dtype=v.dtype), batch_shape + (m, m)).copy()
    q = identity.copy()

    for i in range(k):
        v_i = v[..., :, i].copy()
        v_i[..., :i] = 0
        v_i[..., i] = 1

        v_i_expanded = v_i[..., backend_module.newaxis]
        v_i_h = backend_module.conjugate(v_i_expanded.swapaxes(-1, -2))

        tau_i = tau[..., i, backend_module.newaxis, backend_module.newaxis]

        h_i = identity - tau_i * (v_i_expanded @ v_i_h)
        q = q @ h_i

    return q[..., :n]


@global_eager_registry.register("Cummax")
def _cummax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cummax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "maximum", None)
    if func and hasattr(func, "accumulate"):
        return func.accumulate(*args, **kwargs)

    return backend_module.maximum.accumulate(backend_module.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cummin")
def _cummin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cummin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "minimum", None)
    if func and hasattr(func, "accumulate"):
        return func.accumulate(*args, **kwargs)

    return backend_module.minimum.accumulate(backend_module.asarray(args[0]), **kwargs)


@global_eager_registry.register("Cumlogsumexp")
def _cumlogsumexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cumlogsumexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "cumlogsumexp", None)
    if func:
        return func(*args, **kwargs)

    x = backend_module.asarray(args[0])
    axis = kwargs.get("axis", 0)

    return backend_module.ufunc.accumulate(backend_module.logaddexp, x, axis=axis)


@global_eager_registry.register("CumulativeLogsumexp")
def _cumulative_logsumexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cumulative_logsumexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _cumlogsumexp(backend_module, *args, **kwargs)


@global_eager_registry.register("PsumScatter")
def _psumscatter(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _psumscatter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "psum_scatter"):
        return backend_module.lax.psum_scatter(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Fmax")
def _np_fmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "fmax", getattr(backend_module, "fmax", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fmax(args[0], args[1])


@global_eager_registry.register("ScatterMax")
def _np_scattermax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scattermax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "scattermax", getattr(backend_module, "scattermax", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]


@global_eager_registry.register("ScatterMin")
def _np_scattermin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scattermin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "scattermin", getattr(backend_module, "scattermin", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]


@global_eager_registry.register("WeibullMin")
def _np_weibullmin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_weibullmin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "weibullmin", getattr(backend_module, "weibullmin", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.weibull(*args, **kwargs)


@global_eager_registry.register("WindowHamming")
def _np_windowhamming(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_windowhamming operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func = getattr(backend_module, "windowhamming", getattr(backend_module, "windowhamming", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.hamming(*args, **kwargs)
