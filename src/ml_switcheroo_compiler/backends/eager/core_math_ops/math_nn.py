"""math_nn module."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Sequence
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(backend_module: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager activity regularization operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return x


@global_eager_registry.register("LayerNorm")
def _layer_norm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate LayerNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: Normalized output.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    gamma = cast(EagerTensorProtocol, args[1]) if len(args) > 1 else 1.0
    beta = cast(EagerTensorProtocol, args[2]) if len(args) > 2 else 0.0
    mean = cast(EagerTensorProtocol, mod.mean(x, axis=-1, keepdims=True))
    var = cast(EagerTensorProtocol, mod.var(x, axis=-1, keepdims=True))
    norm = (x - mean) / cast(EagerTensorProtocol, mod.sqrt(var + 1e-5))
    return norm * gamma + beta


@global_eager_registry.register("BatchNorm")
def _batch_norm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate BatchNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional), mean (optional), var (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: Normalized output.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    gamma = cast(EagerTensorProtocol, args[1] if len(args) > 1 else kwargs.get("gamma", 1.0))
    beta = cast(EagerTensorProtocol, args[2] if len(args) > 2 else kwargs.get("beta", 0.0))
    eps = cast(EagerTensorProtocol, kwargs.get("eps", 1e-5))
    if len(args) > 4:
        mean = cast(EagerTensorProtocol, args[3])
        var = cast(EagerTensorProtocol, args[4])
    else:
        axes = tuple(i for i in range(x.ndim) if i != 1) if getattr(x, "ndim", 0) > 1 else (0,)
        mean = cast(EagerTensorProtocol, mod.mean(x, axis=axes, keepdims=True))
        var = cast(EagerTensorProtocol, mod.var(x, axis=axes, keepdims=True))
    norm = (x - mean) / cast(EagerTensorProtocol, mod.sqrt(var + eps))
    return norm * gamma + beta


@global_eager_registry.register("RMSNorm")
def _rms_norm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate RMSNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: Normalized output.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    gamma = cast(EagerTensorProtocol, args[1]) if len(args) > 1 else 1.0
    rms = cast(EagerTensorProtocol, mod.sqrt(cast(EagerTensorProtocol, mod.mean(x**2, axis=-1, keepdims=True)) + 1e-5))
    return (x / rms) * gamma


@global_eager_registry.register("MSELoss")
def _mse_loss(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Mean Squared Error loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: MSE loss value.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    pred = cast(EagerTensorProtocol, args[0])
    tgt = cast(EagerTensorProtocol, args[1])
    return cast(EagerValue, mod.mean((pred - tgt) ** 2))


@global_eager_registry.register("BCELoss")
def _bce_loss(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Binary Cross Entropy loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: BCE loss value.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    pred = cast(EagerTensorProtocol, args[0])
    tgt = cast(EagerTensorProtocol, args[1])
    eps = 1e-7
    pred_c = cast(EagerTensorProtocol, mod.clip(pred, eps, 1.0 - eps)) if hasattr(mod, "clip") else pred
    return -cast(
        EagerTensorProtocol,
        mod.mean(tgt * cast(EagerTensorProtocol, mod.log(pred_c)) + (1.0 - tgt) * cast(EagerTensorProtocol, mod.log(1.0 - pred_c))),
    )


@global_eager_registry.register("HuberLoss")
def _huber_loss(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Huber loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: Huber loss value.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    pred = cast(EagerTensorProtocol, args[0])
    tgt = cast(EagerTensorProtocol, args[1])
    diff = cast(EagerTensorProtocol, mod.abs(pred - tgt))
    cond = diff < 1.0
    loss = mod.where(cond, 0.5 * (diff**2), diff - 0.5)
    return cast(EagerValue, mod.mean(loss))


@global_eager_registry.register("KLDivergenceLoss")
def _kl_loss(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Kullback-Leibler divergence loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: KL divergence loss value.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    pred = cast(EagerTensorProtocol, args[0])
    tgt = cast(EagerTensorProtocol, args[1])
    eps = 1e-7
    return cast(EagerValue, mod.mean(tgt * (cast(EagerTensorProtocol, mod.log(tgt + eps)) - cast(EagerTensorProtocol, mod.log(pred + eps)))))


@global_eager_registry.register("GELU")
@global_eager_registry.register("Gelu")
def _gelu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate GELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: GELU activation output.
    """
    del kwargs
    import math

    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    return 0.5 * x * (1.0 + cast(EagerTensorProtocol, mod.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3)))))


@global_eager_registry.register("SiLU")
@global_eager_registry.register("Silu")
def _silu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate SiLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: SiLU activation output.
    """
    del kwargs
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    sig = 1.0 / (1.0 + cast(EagerTensorProtocol, mod.exp(-x)))
    return x * sig


@global_eager_registry.register("ELU")
@global_eager_registry.register("Elu")
def _elu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: ELU activation output.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    alpha = cast(EagerTensorProtocol, kwargs.get("alpha", 1.0))
    cond = x > 0
    return cast(EagerValue, mod.where(cond, x, alpha * (cast(EagerTensorProtocol, mod.exp(x)) - 1.0)))


@global_eager_registry.register("LeakyReLU")
@global_eager_registry.register("LeakyRelu")
def _leaky_relu(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate LeakyReLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        EagerValue: LeakyReLU activation output.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    alpha = cast(EagerTensorProtocol, kwargs.get("negative_slope", 0.01))
    cond = x > 0
    return cast(EagerValue, mod.where(cond, x, alpha * x))


def _adaptive_pool_1d(backend_module: EagerValue, operand: EagerValue, in_w: int, out_w: int) -> EagerValue:
    """Compute 1D adaptive average pooling over the last axis.

    Args:
        backend_module (EagerValue): Backend execution module.
        operand (EagerValue): Input tensor.
        in_w (int): Input spatial width.
        out_w (int): Target output spatial width.

    Returns:
        EagerValue: Pooled 1D tensor.
    """
    import math

    mod = cast(ModuleType, backend_module)
    bins = []
    for j in range(out_w):
        start = math.floor(j * in_w / out_w)
        end = math.ceil((j + 1) * in_w / out_w)
        bins.append(mod.mean(cast(EagerTensorProtocol, operand)[..., start:end], axis=-1))
    return cast(EagerValue, mod.stack(bins, axis=-1))


def _adaptive_pool_2d(
    backend_module: EagerValue,
    operand: EagerValue,
    in_spatial: tuple[int, int],
    out_spatial: list[int],
) -> EagerValue:
    """Compute 2D adaptive average pooling over the last two axes.

    Args:
        backend_module (EagerValue): Backend execution module.
        operand (EagerValue): Input tensor.
        in_spatial (tuple[int, int]): Input (height, width).
        out_spatial (list[int]): Target output [height, width].

    Returns:
        EagerValue: Pooled 2D tensor.
    """
    import math

    mod = cast(ModuleType, backend_module)
    o_h, o_w = out_spatial
    i_h, i_w = in_spatial
    rows = []
    for i in range(o_h):
        h_start = math.floor(i * i_h / o_h)
        h_end = math.ceil((i + 1) * i_h / o_h)
        cols = []
        for j in range(o_w):
            w_start = math.floor(j * i_w / o_w)
            w_end = math.ceil((j + 1) * i_w / o_w)
            cols.append(mod.mean(cast(EagerTensorProtocol, operand)[..., h_start:h_end, w_start:w_end], axis=(-2, -1)))
        rows.append(mod.stack(cols, axis=-1))
    return cast(EagerValue, mod.stack(rows, axis=-2))


def _adaptive_pool_3d(
    backend_module: EagerValue,
    operand: EagerValue,
    in_spatial: tuple[int, int, int],
    out_spatial: list[int],
) -> EagerValue:
    """Compute 3D adaptive average pooling over the last three axes.

    Args:
        backend_module (EagerValue): Backend execution module.
        operand (EagerValue): Input tensor.
        in_spatial (tuple[int, int, int]): Input (depth, height, width).
        out_spatial (list[int]): Target output [depth, height, width].

    Returns:
        EagerValue: Pooled 3D tensor.
    """
    import math

    mod = cast(ModuleType, backend_module)
    o_d, o_h, o_w = out_spatial
    i_d, i_h, i_w = in_spatial
    depths = []
    for k in range(o_d):
        d_start = math.floor(k * i_d / o_d)
        d_end = math.ceil((k + 1) * i_d / o_d)
        rows = []
        for i in range(o_h):
            h_start = math.floor(i * i_h / o_h)
            h_end = math.ceil((i + 1) * i_h / o_h)
            cols = []
            for j in range(o_w):
                w_start = math.floor(j * i_w / o_w)
                w_end = math.ceil((j + 1) * i_w / o_w)
                cols.append(mod.mean(cast(EagerTensorProtocol, operand)[..., d_start:d_end, h_start:h_end, w_start:w_end], axis=(-3, -2, -1)))
            rows.append(mod.stack(cols, axis=-1))
        depths.append(mod.stack(rows, axis=-2))
    return cast(EagerValue, mod.stack(depths, axis=-3))


def _global_adaptive_pool(backend_module: EagerValue, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _global_adaptive_pool operation rigorously over spatial dimensions.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Result.
    """
    del kwargs
    if not hasattr(operand, "shape"):
        return operand

    out_spatial = [output_size] if isinstance(output_size, int) else list(cast(Sequence[int], output_size))
    spatial_dims = len(out_spatial)
    in_shape = cast(EagerTensorProtocol, operand).shape
    if len(in_shape) < spatial_dims:
        return operand
    in_spatial = in_shape[-spatial_dims:]

    if spatial_dims == 1:
        return _adaptive_pool_1d(backend_module, operand, in_spatial[0], out_spatial[0])
    if spatial_dims == 2:
        return _adaptive_pool_2d(backend_module, operand, (in_spatial[0], in_spatial[1]), out_spatial)
    if spatial_dims == 3:
        return _adaptive_pool_3d(backend_module, operand, (in_spatial[0], in_spatial[1], in_spatial[2]), out_spatial)
    return operand


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(backend_module: EagerValue, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive avg pool2d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(backend_module: EagerValue, operand: EagerValue, output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager adaptive avg pool3d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        operand: The operand parameter.
        output_size: The output_size parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(backend_module: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager alpha dropout operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return x


@global_eager_registry.register("FractionalAvgPool")
def _np_fractionalavgpool(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fractionalavgpool operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fractionalavgpool", getattr(backend_module, "fractionalavgpool", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]
