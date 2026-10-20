"""Implementation of nn ops operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.nn_ops`)."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("Dropout2d")
def _np_dropout2d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dropout2d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    p = kwargs.get("p", 0.5)
    training = kwargs.get("training", True)
    if not training or p == 0.0:
        return x

    # dropout2d zeros out entire channels (axis 1) for each sample in batch (axis 0)
    # Shape of x: (N, C, H, W)
    if getattr(x, "ndim") != 4:
        raise ValueError("Dropout2d requires a 4D tensor (N, C, H, W)")

    N, C, _, _ = getattr(x, "shape")
    mask = getattr(getattr(np.random, "binomial")(1, 1.0 - typing.cast(EagerTensorProtocol, p), size=(N, C, 1, 1)), "astype")(getattr(x, "dtype"))
    return typing.cast("EagerValue", typing.cast(EagerTensorProtocol, x) * mask / (1.0 - typing.cast(EagerTensorProtocol, p)))


@numpy_eager_registry.register("BlockMaskedMm")
def _np_block_masked_mm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np block masked mm operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    a, b = args[0], args[1]
    # BlockMaskedMm is just batched matmul where some blocks are skipped.
    # In eager mode without a provided mask, we just return a @ b
    return typing.cast("EagerValue", getattr(np, "matmul")(a, b))


@numpy_eager_registry.register("Mul")
def _np_mul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate elementwise multiplication in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Input operands.
        **kwargs (EagerValue): Optional kwargs.

    Returns:
        np.ndarray: Multiplied product array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np, "multiply")(args[0], args[1]))


@numpy_eager_registry.register("MaxPool2DWithArgmax")
def _np_max_pool2d_with_argmax_forward(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate 2D max pooling forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Input tensor [x].
        **kwargs (EagerValue): Kernel and stride specs.

    Returns:
        np.ndarray: Max pooled output.
    """
    x = np.asarray(args[0])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_h, k_w = (k, k) if isinstance(k, int) else (getattr(k, "__getitem__")(0), getattr(k, "__getitem__")(1))
    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_h, s_w = (s, s) if isinstance(s, int) else (getattr(s, "__getitem__")(0), getattr(s, "__getitem__")(1))
    n, c, h, w = x.shape
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            typing.cast(EagerTensorProtocol, out)[:, :, i, j] = getattr(np, "max")(typing.cast(EagerTensorProtocol, x)[:, :, h_start:h_end, w_start:w_end], axis=(-2, -1))
    return out


@numpy_eager_registry.register("BatchNorm")
def _np_batch_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate BatchNorm forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Positional inputs [x, gamma, beta].
        **kwargs (EagerValue): Keyword arguments (eps, epsilon).

    Returns:
        np.ndarray: Normalized output tensor.
    """
    x = np.asarray(args[0])
    c_dim = x.shape[1] if x.ndim > 1 else x.shape[0]
    gamma = np.asarray(args[1]) if len(args) > 1 and args[1] is not None else np.ones(c_dim, dtype=x.dtype)
    beta = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.zeros(c_dim, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    broadcast_shape = [1] * x.ndim
    if x.ndim > 1:
        broadcast_shape[1] = gamma.size
    else:
        broadcast_shape[0] = gamma.size

    g = gamma.reshape(broadcast_shape)
    b = beta.reshape(broadcast_shape)

    mean = np.mean(x, axis=reduction_axes, keepdims=True)
    var = np.var(x, axis=reduction_axes, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], g * x_hat + b)


@numpy_eager_registry.register("BatchNormInputGrad")
def _np_batch_norm_input_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact analytical gradient of BatchNorm with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x, gamma].
        **kwargs (EagerValue): Keyword arguments (eps).

    Returns:
        np.ndarray: Input gradient tensor matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    c_dim = x.shape[1] if x.ndim > 1 else x.shape[0]
    gamma = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.ones(c_dim, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    broadcast_shape = [1] * x.ndim
    if x.ndim > 1:
        broadcast_shape[1] = gamma.size
    else:
        broadcast_shape[0] = gamma.size

    g = gamma.reshape(broadcast_shape)
    mean = np.mean(x, axis=reduction_axes, keepdims=True)
    var = np.var(x, axis=reduction_axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    m = float(np.prod([x.shape[i] for i in reduction_axes]))
    d_beta = np.sum(d_out, axis=reduction_axes, keepdims=True)
    d_gamma = np.sum(d_out * x_hat, axis=reduction_axes, keepdims=True)

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], (g * inv_std / m) * (m * d_out - d_beta - x_hat * d_gamma))


@numpy_eager_registry.register("BatchNormGammaGrad")
def _np_batch_norm_gamma_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of BatchNorm with respect to scale (gamma).

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Keyword arguments (eps).

    Returns:
        np.ndarray: Gamma gradient vector.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    mean = np.mean(x, axis=reduction_axes, keepdims=True)
    var = np.var(x, axis=reduction_axes, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out * x_hat, axis=reduction_axes).flatten())


@numpy_eager_registry.register("BatchNormBetaGrad")
def _np_batch_norm_beta_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of BatchNorm with respect to shift (beta).

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out].
        **kwargs (EagerValue): Optional keyword arguments.

    Returns:
        np.ndarray: Beta gradient vector.
    """
    d_out = np.asarray(args[0])
    reduction_axes = tuple(i for i in range(d_out.ndim) if i != 1) if d_out.ndim > 1 else (0,)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out, axis=reduction_axes).flatten())


@numpy_eager_registry.register("LayerNorm")
def _np_layer_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate LayerNorm forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Positional arguments [x, gamma, beta].
        **kwargs (EagerValue): Keyword arguments (eps, axis).

    Returns:
        np.ndarray: Layer-normalized output.
    """
    x = np.asarray(args[0])
    last_dim = x.shape[-1]
    gamma = np.asarray(args[1]) if len(args) > 1 and args[1] is not None else np.ones(last_dim, dtype=x.dtype)
    beta = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.zeros(last_dim, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    mean = getattr(np, "mean")(x, axis=axis, keepdims=True)
    var = getattr(np, "var")(x, axis=axis, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], gamma * x_hat + beta)


@numpy_eager_registry.register("LayerNormInputGrad")
def _np_layer_norm_input_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact analytical gradient of LayerNorm with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x, gamma].
        **kwargs (EagerValue): Keyword arguments (eps, axis).

    Returns:
        np.ndarray: Input gradient matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    gamma = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.ones(x.shape[-1], dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    d = float(getattr(x.shape, "__getitem__")(axis))
    mean = getattr(np, "mean")(x, axis=axis, keepdims=True)
    var = getattr(np, "var")(x, axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    d_y = d_out * gamma
    term1 = d * d_y
    term2 = getattr(np, "sum")(d_y, axis=axis, keepdims=True)
    term3 = x_hat * getattr(np, "sum")(d_y * x_hat, axis=axis, keepdims=True)

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], (inv_std / d) * (term1 - term2 - term3))


@numpy_eager_registry.register("LayerNormGammaGrad")
def _np_layer_norm_gamma_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of LayerNorm with respect to gamma.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Keyword arguments (eps, axis).

    Returns:
        np.ndarray: Gamma gradient.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    mean = getattr(np, "mean")(x, axis=axis, keepdims=True)
    var = getattr(np, "var")(x, axis=axis, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    batch_axes = tuple(i for i in range(x.ndim) if i != (x.ndim + int(typing.cast(int, axis)) if int(typing.cast(int, axis)) < 0 else int(typing.cast(int, axis))))
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out * x_hat, axis=batch_axes))


@numpy_eager_registry.register("LayerNormBetaGrad")
def _np_layer_norm_beta_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of LayerNorm with respect to beta.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out].
        **kwargs (EagerValue): Keyword arguments (axis).

    Returns:
        np.ndarray: Beta gradient.
    """
    d_out = np.asarray(args[0])
    axis = kwargs.get("axis", -1)
    batch_axes = tuple(i for i in range(d_out.ndim) if i != (d_out.ndim + int(typing.cast(int, axis)) if int(typing.cast(int, axis)) < 0 else int(typing.cast(int, axis))))
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out, axis=batch_axes))


@numpy_eager_registry.register("RMSNorm")
def _np_rms_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate RMSNorm forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Positional arguments [x, gamma].
        **kwargs (EagerValue): Keyword arguments (eps).

    Returns:
        np.ndarray: Normalized tensor.
    """
    x = np.asarray(args[0])
    last_dim = x.shape[-1]
    gamma = np.asarray(args[1]) if len(args) > 1 and args[1] is not None else np.ones(last_dim, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], gamma * x_hat)


@numpy_eager_registry.register("RMSNormInputGrad")
def _np_rms_norm_input_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact analytical gradient of RMSNorm with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x, gamma].
        **kwargs (EagerValue): Keyword arguments (eps).

    Returns:
        np.ndarray: Input gradient matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    gamma = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.ones(x.shape[-1], dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    d = float(x.shape[-1])
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    d_y = d_out * gamma

    term = x_hat * np.sum(d_y * x_hat, axis=-1, keepdims=True)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], (1.0 / rms) * (d_y - (1.0 / d) * term))


@numpy_eager_registry.register("RMSNormGammaGrad")
def _np_rms_norm_gamma_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of RMSNorm with respect to gamma.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Keyword arguments (eps).

    Returns:
        np.ndarray: Gamma gradient vector.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    batch_axes = tuple(range(x.ndim - 1))
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out * x_hat, axis=batch_axes))


@numpy_eager_registry.register("GroupNorm")
def _np_group_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate GroupNorm forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [x, gamma, beta].
        **kwargs (EagerValue): Keyword arguments (num_groups, groups, eps).

    Returns:
        np.ndarray: Group-normalized output.
    """
    x = np.asarray(args[0])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(typing.cast(int, kwargs.get("num_groups", kwargs.get("groups", 1))))
    gamma = np.asarray(args[1]) if len(args) > 1 and args[1] is not None else np.ones(c, dtype=x.dtype)
    beta = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.zeros(c, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    mean = np.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = np.var(x_grouped, axis=norm_axes, keepdims=True)
    x_hat = (x_grouped - mean) / np.sqrt(var + eps)
    x_hat_flat = x_hat.reshape(x.shape)

    b_shape = [1] * x.ndim
    b_shape[1] = c
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], gamma.reshape(b_shape) * x_hat_flat + beta.reshape(b_shape))


@numpy_eager_registry.register("GroupNormInputGrad")
def _np_group_norm_input_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of GroupNorm with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x, gamma].
        **kwargs (EagerValue): Keyword arguments (num_groups, groups, eps).

    Returns:
        np.ndarray: Input gradient matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(typing.cast(int, kwargs.get("num_groups", kwargs.get("groups", 1))))
    gamma = np.asarray(args[2]) if len(args) > 2 and args[2] is not None else np.ones(c, dtype=x.dtype)
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    b_shape = [1] * x.ndim
    b_shape[1] = c
    d_out_scaled = d_out * gamma.reshape(b_shape)
    d_out_grouped = d_out_scaled.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    m = float(np.prod([x_grouped.shape[i] for i in norm_axes]))
    mean = np.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = np.var(x_grouped, axis=norm_axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x_grouped - mean) * inv_std

    d_beta = np.sum(d_out_grouped, axis=norm_axes, keepdims=True)
    d_gamma = np.sum(d_out_grouped * x_hat, axis=norm_axes, keepdims=True)

    dx_grouped = (inv_std / m) * (m * d_out_grouped - d_beta - x_hat * d_gamma)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], dx_grouped.reshape(x.shape))


@numpy_eager_registry.register("GroupNormGammaGrad")
def _np_group_norm_gamma_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of GroupNorm with respect to gamma.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Keyword arguments (num_groups, groups, eps).

    Returns:
        np.ndarray: Gamma gradient vector.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(typing.cast(int, kwargs.get("num_groups", kwargs.get("groups", 1))))
    eps = float(typing.cast(float, kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    mean = np.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = np.var(x_grouped, axis=norm_axes, keepdims=True)
    x_hat = (x_grouped - mean) / np.sqrt(var + eps)
    x_hat_flat = x_hat.reshape(x.shape)

    axes_to_sum = (0,) + tuple(range(2, x.ndim))
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out * x_hat_flat, axis=axes_to_sum))


@numpy_eager_registry.register("GroupNormBetaGrad")
def _np_group_norm_beta_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of GroupNorm with respect to beta.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out].
        **kwargs (EagerValue): Optional keyword arguments.

    Returns:
        np.ndarray: Beta gradient vector.
    """
    d_out = np.asarray(args[0])
    axes_to_sum = (0,) + tuple(range(2, d_out.ndim))
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.sum(d_out, axis=axes_to_sum))


@numpy_eager_registry.register("AvgPool2D")
def _np_avg_pool2d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate AvgPool2D forward in NumPy eager mode.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): Positional args [x].
        **kwargs (EagerValue): Kernel and stride specs.

    Returns:
        np.ndarray: Average-pooled output tensor.
    """
    x = np.asarray(args[0])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_h, k_w = (k, k) if isinstance(k, int) else (getattr(k, "__getitem__")(0), getattr(k, "__getitem__")(1))

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_h, s_w = (s, s) if isinstance(s, int) else (getattr(s, "__getitem__")(0), getattr(s, "__getitem__")(1))

    n, c, h, w = x.shape
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            typing.cast(EagerTensorProtocol, out)[:, :, i, j] = getattr(np, "mean")(typing.cast(EagerTensorProtocol, x)[:, :, h_start:h_end, w_start:w_end], axis=(-2, -1))
    return out


@numpy_eager_registry.register("AvgPool2DGrad")
def _np_avg_pool2d_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of AvgPool2D with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Kernel and stride specs.

    Returns:
        np.ndarray: Input gradient matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_h, k_w = (k, k) if isinstance(k, int) else (getattr(k, "__getitem__")(0), getattr(k, "__getitem__")(1))

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_h, s_w = (s, s) if isinstance(s, int) else (getattr(s, "__getitem__")(0), getattr(s, "__getitem__")(1))

    dx = np.zeros_like(x)
    out_h, out_w = d_out.shape[2], d_out.shape[3]
    scale = 1.0 / float(k_h * k_w)

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            typing.cast(EagerTensorProtocol, dx)[:, :, h_start:h_end, w_start:w_end] += typing.cast(EagerTensorProtocol, d_out)[:, :, i : i + 1, j : j + 1] * scale
    return dx


@numpy_eager_registry.register("MaxPool2DWithArgmaxGrad")
def _np_max_pool2d_with_argmax_grad(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exact gradient of MaxPool2DWithArgmax with respect to input.

    Args:
        backend_module (ModuleType): Backend provider module.
        *args (EagerValue): [d_out, x].
        **kwargs (EagerValue): Kernel and stride specs.

    Returns:
        np.ndarray: Input gradient matching x shape.
    """
    d_out = np.asarray(args[0])
    x = np.asarray(args[1])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_h, k_w = (k, k) if isinstance(k, int) else (getattr(k, "__getitem__")(0), getattr(k, "__getitem__")(1))

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_h, s_w = (s, s) if isinstance(s, int) else (getattr(s, "__getitem__")(0), getattr(s, "__getitem__")(1))

    dx = np.zeros_like(x)
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            window = typing.cast(EagerTensorProtocol, x)[:, :, h_start:h_end, w_start:w_end]
            max_val = getattr(np, "max")(window, axis=(-2, -1), keepdims=True)
            mask = (window == max_val).astype(x.dtype)
            mask_sum = np.sum(mask, axis=(-2, -1), keepdims=True)
            mask = mask / np.maximum(mask_sum, 1.0)
            typing.cast(EagerTensorProtocol, dx)[:, :, h_start:h_end, w_start:w_end] += typing.cast(EagerTensorProtocol, d_out)[:, :, i : i + 1, j : j + 1] * mask
    return dx
