# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_nn module."""

from __future__ import annotations

import builtins
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(backend_module: Any, x: object, **kwargs: Any) -> Any:
    """Evaluate _activity_regularization operation.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return x


@global_eager_registry.register("LayerNorm")
def _layer_norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate LayerNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        Any: Normalized output.
    """
    del kwargs
    x = args[0]
    gamma = args[1] if len(args) > 1 else 1.0
    beta = args[2] if len(args) > 2 else 0.0
    mean = backend_module.mean(x, axis=-1, keepdims=True)
    var = backend_module.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / backend_module.sqrt(var + 1e-5)
    return norm * gamma + beta


@global_eager_registry.register("BatchNorm")
def _batch_norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate BatchNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional), mean (optional), var (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        Any: Normalized output.
    """
    x = args[0]
    gamma = args[1] if len(args) > 1 else kwargs.get("gamma", 1.0)
    beta = args[2] if len(args) > 2 else kwargs.get("beta", 0.0)
    eps = kwargs.get("eps", 1e-5)
    if len(args) > 4:
        mean = args[3]
        var = args[4]
    else:
        axes = tuple(i for i in range(x.ndim) if i != 1) if getattr(x, "ndim", 0) > 1 else (0,)
        mean = backend_module.mean(x, axis=axes, keepdims=True)
        var = backend_module.var(x, axis=axes, keepdims=True)
    norm = (x - mean) / backend_module.sqrt(var + eps)
    return norm * gamma + beta


@global_eager_registry.register("RMSNorm")
def _rms_norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate RMSNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        Any: Normalized output.
    """
    del kwargs
    x = args[0]
    gamma = args[1] if len(args) > 1 else 1.0
    rms = backend_module.sqrt(backend_module.mean(x**2, axis=-1, keepdims=True) + 1e-5)
    return (x / rms) * gamma


@global_eager_registry.register("MSELoss")
def _mse_loss(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate Mean Squared Error loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: MSE loss value.
    """
    del kwargs
    pred = args[0]
    tgt = args[1]
    return backend_module.mean((pred - tgt) ** 2)


@global_eager_registry.register("BCELoss")
def _bce_loss(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate Binary Cross Entropy loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: BCE loss value.
    """
    del kwargs
    pred = args[0]
    tgt = args[1]
    eps = 1e-7
    pred_c = backend_module.clip(pred, eps, 1.0 - eps) if hasattr(backend_module, "clip") else pred
    return -backend_module.mean(tgt * backend_module.log(pred_c) + (1.0 - tgt) * backend_module.log(1.0 - pred_c))


@global_eager_registry.register("HuberLoss")
def _huber_loss(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate Huber loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: Huber loss value.
    """
    del kwargs
    pred = args[0]
    tgt = args[1]
    diff = backend_module.abs(pred - tgt)
    cond = diff < 1.0
    loss = backend_module.where(cond, 0.5 * (diff**2), diff - 0.5)
    return backend_module.mean(loss)


@global_eager_registry.register("KLDivergenceLoss")
def _kl_loss(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate Kullback-Leibler divergence loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: KL divergence loss value.
    """
    del kwargs
    pred = args[0]
    tgt = args[1]
    eps = 1e-7
    return backend_module.mean(tgt * (backend_module.log(tgt + eps) - backend_module.log(pred + eps)))


@global_eager_registry.register("GELU")
@global_eager_registry.register("Gelu")
def _gelu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate GELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: GELU activation output.
    """
    del kwargs
    import math

    x = args[0]
    return 0.5 * x * (1.0 + backend_module.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


@global_eager_registry.register("SiLU")
@global_eager_registry.register("Silu")
def _silu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate SiLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: SiLU activation output.
    """
    del kwargs
    x = args[0]
    sig = 1.0 / (1.0 + backend_module.exp(-x))
    return x * sig


@global_eager_registry.register("ELU")
@global_eager_registry.register("Elu")
def _elu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate ELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: ELU activation output.
    """
    x = args[0]
    alpha = kwargs.get("alpha", 1.0)
    cond = x > 0
    return backend_module.where(cond, x, alpha * (backend_module.exp(x) - 1.0))


@global_eager_registry.register("LeakyReLU")
@global_eager_registry.register("LeakyRelu")
def _leaky_relu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate LeakyReLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Any: LeakyReLU activation output.
    """
    x = args[0]
    alpha = kwargs.get("negative_slope", 0.01)
    cond = x > 0
    return backend_module.where(cond, x, alpha * x)


def _global_adaptive_pool(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _global_adaptive_pool operation rigorously over spatial dimensions.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    import math

    if not hasattr(operand, "shape"):
        return operand

    if isinstance(output_size, int):
        out_spatial = [output_size]
    else:
        out_spatial = list(output_size)

    spatial_dims = len(out_spatial)
    in_shape = operand.shape
    if len(in_shape) < spatial_dims:
        return operand
    in_spatial = in_shape[-spatial_dims:]

    if spatial_dims == 1:
        O_w = out_spatial[0]
        I_w = in_spatial[0]
        bins = []
        for j in range(O_w):
            start = math.floor(j * I_w / O_w)
            end = math.ceil((j + 1) * I_w / O_w)
            bins.append(backend_module.mean(operand[..., start:end], axis=-1))
        return backend_module.stack(bins, axis=-1)

    if spatial_dims == 2:
        O_h, O_w = out_spatial
        I_h, I_w = in_spatial
        rows = []
        for i in range(O_h):
            h_start = math.floor(i * I_h / O_h)
            h_end = math.ceil((i + 1) * I_h / O_h)
            cols = []
            for j in range(O_w):
                w_start = math.floor(j * I_w / O_w)
                w_end = math.ceil((j + 1) * I_w / O_w)
                cols.append(backend_module.mean(operand[..., h_start:h_end, w_start:w_end], axis=(-2, -1)))
            rows.append(backend_module.stack(cols, axis=-1))
        return backend_module.stack(rows, axis=-2)

    if spatial_dims == 3:
        O_d, O_h, O_w = out_spatial
        I_d, I_h, I_w = in_spatial
        depths = []
        for k in range(O_d):
            d_start = math.floor(k * I_d / O_d)
            d_end = math.ceil((k + 1) * I_d / O_d)
            rows = []
            for i in range(O_h):
                h_start = math.floor(i * I_h / O_h)
                h_end = math.ceil((i + 1) * I_h / O_h)
                cols = []
                for j in range(O_w):
                    w_start = math.floor(j * I_w / O_w)
                    w_end = math.ceil((j + 1) * I_w / O_w)
                    cols.append(backend_module.mean(operand[..., d_start:d_end, h_start:h_end, w_start:w_end], axis=(-3, -2, -1)))
                rows.append(backend_module.stack(cols, axis=-1))
            depths.append(backend_module.stack(rows, axis=-2))
        return backend_module.stack(depths, axis=-3)

    return operand


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_avg_pool2d operation.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_avg_pool3d operation.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(backend_module: Any, x: object, **kwargs: Any) -> Any:
    """Evaluate _alpha_dropout operation.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return x


@global_eager_registry.register("FractionalAvgPool")
def _np_fractionalavgpool(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fractionalavgpool operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "fractionalavgpool", getattr(backend_module, "fractionalavgpool", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]
