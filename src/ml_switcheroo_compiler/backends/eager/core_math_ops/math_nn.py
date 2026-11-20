"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_nn`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    coerce_array,
    coerce_float,
    coerce_numeric,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager activity regularization operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        x: Input `x` argument (x).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return x


@global_eager_registry.register("LayerNorm")
def _layer_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate LayerNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: Normalized output.
    """
    del kwargs
    mod = backend_module
    x = coerce_array(mod, args[0])
    gamma = coerce_numeric(args[1]) if len(args) > 1 else 1.0
    beta = coerce_numeric(args[2]) if len(args) > 2 else 0.0
    mean = adapt_backend_module(mod).require_symbol("mean")(x, axis=-1, keepdims=True)
    var = adapt_backend_module(mod).require_symbol("var")(x, axis=-1, keepdims=True)
    norm = (x - mean) / adapt_backend_module(mod).require_symbol("sqrt")(var + 1e-5)
    return norm * gamma + beta


@global_eager_registry.register("BatchNorm")
def _batch_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate BatchNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional), beta (optional), mean (optional), var (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: Normalized output.
    """
    mod = backend_module
    x = coerce_array(mod, args[0])
    gamma = coerce_numeric(args[1]) if len(args) > 1 else coerce_numeric(kwargs.get("gamma", 1.0))
    beta = coerce_numeric(args[2]) if len(args) > 2 else coerce_numeric(kwargs.get("beta", 0.0))
    eps = coerce_float(kwargs.get("eps", 1e-5))
    if len(args) > 4:
        mean = coerce_numeric(args[3])
        var = coerce_numeric(args[4])
    else:
        axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
        mean = adapt_backend_module(mod).require_symbol("mean")(x, axis=axes, keepdims=True)
        var = adapt_backend_module(mod).require_symbol("var")(x, axis=axes, keepdims=True)
    norm = (x - mean) / adapt_backend_module(mod).require_symbol("sqrt")(var + eps)
    return norm * gamma + beta


@global_eager_registry.register("RMSNorm")
def _rms_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate RMSNorm.

    Args:
        backend_module: Backend execution module.
        *args: Input, gamma (optional).
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: Normalized output.
    """
    del kwargs
    mod = backend_module
    x = coerce_array(mod, args[0])
    gamma = coerce_numeric(args[1]) if len(args) > 1 else 1.0
    rms = adapt_backend_module(mod).require_symbol("sqrt")(
        adapt_backend_module(mod).require_symbol("mean")(x**2, axis=-1, keepdims=True) + 1e-5
    )
    return (x / rms) * gamma


@global_eager_registry.register("MSELoss")
def _mse_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Mean Squared Error loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: MSE loss value.
    """
    del kwargs
    mod = backend_module
    pred = coerce_array(mod, args[0])
    tgt = coerce_numeric(args[1])
    return adapt_backend_module(mod).require_symbol("mean")((pred - tgt) ** 2)


@global_eager_registry.register("BCELoss")
def _bce_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Binary Cross Entropy loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: BCE loss value.
    """
    del kwargs
    mod = backend_module
    pred = coerce_array(mod, args[0])
    tgt = coerce_numeric(args[1])
    eps = 1e-7
    pred_c = (
        adapt_backend_module(mod).require_symbol("clip")(pred, eps, 1.0 - eps)
        if adapt_backend_module(mod).has_symbol("clip")
        else pred
    )
    return -adapt_backend_module(mod).require_symbol("mean")(
        tgt * adapt_backend_module(mod).require_symbol("log")(pred_c)
        + (1.0 - tgt) * adapt_backend_module(mod).require_symbol("log")(1.0 - pred_c)
    )


@global_eager_registry.register("HuberLoss")
def _huber_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Huber loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: Huber loss value.
    """
    del kwargs
    mod = backend_module
    pred = coerce_array(mod, args[0])
    tgt = coerce_numeric(args[1])
    diff = adapt_backend_module(mod).require_symbol("abs")(pred - tgt)
    cond = diff < 1.0
    loss = adapt_backend_module(mod).require_symbol("where")(cond, 0.5 * (diff**2), diff - 0.5)
    return adapt_backend_module(mod).require_symbol("mean")(loss)


@global_eager_registry.register("KLDivergenceLoss")
def _kl_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Kullback-Leibler divergence loss.

    Args:
        backend_module: Backend execution module.
        *args: Predictions, Targets.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: KL divergence loss value.
    """
    del kwargs
    mod = backend_module
    pred = coerce_numeric(args[0])
    tgt = coerce_numeric(args[1])
    eps = 1e-7
    return adapt_backend_module(mod).require_symbol("mean")(
        tgt
        * (
            adapt_backend_module(mod).require_symbol("log")(tgt + eps)
            - adapt_backend_module(mod).require_symbol("log")(pred + eps)
        )
    )


@global_eager_registry.register("GELU")
@global_eager_registry.register("Gelu")
def _gelu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate GELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: GELU activation output.
    """
    del kwargs
    import math

    mod = backend_module
    x = coerce_numeric(args[0])
    return (
        0.5
        * x
        * (1.0 + adapt_backend_module(mod).require_symbol("tanh")(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))
    )


@global_eager_registry.register("SiLU")
@global_eager_registry.register("Silu")
def _silu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate SiLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: SiLU activation output.
    """
    del kwargs
    mod = backend_module
    x = coerce_numeric(args[0])
    sig = 1.0 / (1.0 + adapt_backend_module(mod).require_symbol("exp")(-x))
    return x * sig


@global_eager_registry.register("ELU")
@global_eager_registry.register("Elu")
def _elu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate ELU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: ELU activation output.
    """
    mod = backend_module
    x = coerce_numeric(args[0])
    alpha = coerce_float(kwargs.get("alpha", 1.0))
    cond = x > 0
    return adapt_backend_module(mod).require_symbol("where")(
        cond, x, alpha * (adapt_backend_module(mod).require_symbol("exp")(x) - 1.0)
    )


@global_eager_registry.register("LeakyReLU")
@global_eager_registry.register("LeakyRelu")
def _leaky_relu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate LeakyReLU activation.

    Args:
        backend_module: Backend execution module.
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        BackendArg: LeakyReLU activation output.
    """
    mod = backend_module
    x = coerce_numeric(args[0])
    alpha = coerce_float(kwargs.get("negative_slope", 0.01))
    cond = x > 0
    return adapt_backend_module(mod).require_symbol("where")(cond, x, alpha * x)


def _adaptive_pool_1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArrayProtocol,
    in_w: int,
    out_w: int,
) -> BackendArg:
    """Compute 1D adaptive average pooling over the last axis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend execution module.
        operand (BackendArrayProtocol): Input tensor.
        in_w (int): Input spatial width.
        out_w (int): Target output spatial width.

    Returns:
        BackendArg: Pooled 1D tensor.
    """
    import math

    mod = backend_module
    bins = []
    for j in range(out_w):
        start = math.floor(j * in_w / out_w)
        end = math.ceil((j + 1) * in_w / out_w)
        bins.append(adapt_backend_module(mod).require_symbol("mean")(operand[..., start:end], axis=-1))
    return adapt_backend_module(mod).require_symbol("stack")(bins, axis=-1)


def _adaptive_pool_2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArrayProtocol,
    in_spatial: tuple[int, int],
    out_spatial: list[int],
) -> BackendArg:
    """Compute 2D adaptive average pooling over the last two axes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend execution module.
        operand (BackendArrayProtocol): Input tensor.
        in_spatial (tuple[int, int]): Input (height, width).
        out_spatial (list[int]): Target output [height, width].

    Returns:
        BackendArg: Pooled 2D tensor.
    """
    import math

    mod = backend_module
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
            cols.append(
                adapt_backend_module(mod).require_symbol("mean")(
                    operand[..., h_start:h_end, w_start:w_end], axis=(-2, -1)
                )
            )
        rows.append(adapt_backend_module(mod).require_symbol("stack")(cols, axis=-1))
    return adapt_backend_module(mod).require_symbol("stack")(rows, axis=-2)


def _adaptive_pool_3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArrayProtocol,
    in_spatial: tuple[int, int, int],
    out_spatial: list[int],
) -> BackendArg:
    """Compute 3D adaptive average pooling over the last three axes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend execution module.
        operand (BackendArrayProtocol): Input tensor.
        in_spatial (tuple[int, int, int]): Input (depth, height, width).
        out_spatial (list[int]): Target output [depth, height, width].

    Returns:
        BackendArg: Pooled 3D tensor.
    """
    import math

    mod = backend_module
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
                cols.append(
                    adapt_backend_module(mod).require_symbol("mean")(
                        operand[..., d_start:d_end, h_start:h_end, w_start:w_end],
                        axis=(-3, -2, -1),
                    )
                )
            rows.append(adapt_backend_module(mod).require_symbol("stack")(cols, axis=-1))
        depths.append(adapt_backend_module(mod).require_symbol("stack")(rows, axis=-2))
    return adapt_backend_module(mod).require_symbol("stack")(depths, axis=-3)


def _global_adaptive_pool(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Evaluate _global_adaptive_pool operation rigorously over spatial dimensions.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    del kwargs
    if not isinstance(operand, BackendArrayProtocol):
        return operand

    out_spatial = [output_size] if isinstance(output_size, int) else list(coerce_shape(output_size))
    spatial_dims = len(out_spatial)
    in_shape = operand.shape
    if len(in_shape) < spatial_dims:
        _typed_ret_2: BackendArg = operand
        return _typed_ret_2
    in_spatial = in_shape[-spatial_dims:]

    if spatial_dims == 1:
        return _adaptive_pool_1d(backend_module, operand, in_spatial[0], out_spatial[0])
    if spatial_dims == 2:
        return _adaptive_pool_2d(backend_module, operand, (in_spatial[0], in_spatial[1]), out_spatial)
    if spatial_dims == 3:
        return _adaptive_pool_3d(backend_module, operand, (in_spatial[0], in_spatial[1], in_spatial[2]), out_spatial)
    _typed_ret_1: BackendArg = operand
    return _typed_ret_1


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive avg pool2d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager adaptive avg pool3d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        operand: Input `operand` argument (operand).
        output_size: Input `output_size` argument (output size).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager alpha dropout operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        x: Input `x` argument (x).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return x


@global_eager_registry.register("FractionalAvgPool")
def _np_fractionalavgpool(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fractionalavgpool operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fractionalavgpool", adapt_backend_module(backend_module).get_symbol("fractionalavgpool", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]
