"""Generic backend-parametric eager reference operations for `nn_ops.layers`."""

from __future__ import annotations

import math
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    coerce_int,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Dropout2d")
def _np_dropout2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np dropout2d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    arr_mod = _resolve_eager_mod(backend_module)
    x = args[0]
    p = coerce_float(kwargs.get("p", 0.5))
    training = kwargs.get("training", True)
    if not training or p == 0.0:
        return x

    # dropout2d zeros out entire channels (axis 1) for each sample in batch (axis 0)
    # Shape of x: (N, C, H, W)
    if adapt_backend_module(x).require_symbol("ndim") != 4:
        raise ValueError("Dropout2d requires a 4D tensor (N, C, H, W)")

    N, C, _, _ = coerce_shape(adapt_backend_module(x).require_symbol("shape"))
    mask = adapt_backend_module(
        adapt_backend_module(arr_mod.random).require_symbol("binomial")(1, 1.0 - p, size=(N, C, 1, 1))
    ).require_symbol("astype")(adapt_backend_module(x).require_symbol("dtype"))
    return coerce_numeric(x) * mask / (1.0 - p)


@global_eager_registry.register("BlockMaskedMm")
def _np_block_masked_mm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np block masked mm operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a, b = args[0], args[1]
    # BlockMaskedMm is just batched matmul where some blocks are skipped.
    # In eager mode without a provided mask, we just return a @ b
    return adapt_backend_module(arr_mod).require_symbol("matmul")(a, b)


@global_eager_registry.register("Mul")
def _np_mul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate elementwise multiplication in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Input operands.
        **kwargs (BackendArg): Optional kwargs.

    Returns:
        EagerTensorProtocol: Multiplied product array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(adapt_backend_module(arr_mod).require_symbol("multiply")(args[0], args[1]))


@global_eager_registry.register("MaxPool2DWithArgmax")
def _np_max_pool2d_with_argmax_forward(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate 2D max pooling forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Input tensor [x].
        **kwargs (BackendArg): Kernel and stride specs.

    Returns:
        EagerTensorProtocol: Max pooled output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_dims = (k, k) if isinstance(k, int) else coerce_shape(k)
    k_h, k_w = k_dims[0], k_dims[1]
    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_dims = (s, s) if isinstance(s, int) else coerce_shape(s)
    s_h, s_w = s_dims[0], s_dims[1]
    n, c, h, w = x.shape
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = arr_mod.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            out[:, :, i, j] = adapt_backend_module(arr_mod).require_symbol("max")(
                x[:, :, h_start:h_end, w_start:w_end], axis=(-2, -1)
            )
    return out


@global_eager_registry.register("BatchNorm")
def _np_batch_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate BatchNorm forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional inputs [x, gamma, beta].
        **kwargs (BackendArg): Keyword arguments (eps, epsilon).

    Returns:
        EagerTensorProtocol: Normalized output tensor.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    c_dim = x.shape[1] if x.ndim > 1 else x.shape[0]
    gamma = arr_mod.asarray(args[1]) if len(args) > 1 and args[1] is not None else arr_mod.ones(c_dim, dtype=x.dtype)
    beta = arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.zeros(c_dim, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    broadcast_shape = [1] * x.ndim
    if x.ndim > 1:
        broadcast_shape[1] = int(coerce_int(gamma.size))
    else:
        broadcast_shape[0] = int(coerce_int(gamma.size))

    g = gamma.reshape(broadcast_shape)
    b = beta.reshape(broadcast_shape)

    mean = arr_mod.mean(x, axis=reduction_axes, keepdims=True)
    var = arr_mod.var(x, axis=reduction_axes, keepdims=True)
    x_hat = (x - mean) / arr_mod.sqrt(var + eps)
    return _ensure_arr(g * x_hat + b)


@global_eager_registry.register("BatchNormInputGrad")
def _np_batch_norm_input_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact analytical gradient of BatchNorm with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x, gamma].
        **kwargs (BackendArg): Keyword arguments (eps).

    Returns:
        EagerTensorProtocol: Input gradient tensor matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    c_dim = x.shape[1] if x.ndim > 1 else x.shape[0]
    gamma = arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.ones(c_dim, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    broadcast_shape = [1] * x.ndim
    if x.ndim > 1:
        broadcast_shape[1] = int(coerce_int(gamma.size))
    else:
        broadcast_shape[0] = int(coerce_int(gamma.size))

    g = gamma.reshape(broadcast_shape)
    mean = arr_mod.mean(x, axis=reduction_axes, keepdims=True)
    var = arr_mod.var(x, axis=reduction_axes, keepdims=True)
    inv_std = 1.0 / arr_mod.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    m = float(arr_mod.prod([x.shape[i] for i in reduction_axes]))
    d_beta = arr_mod.sum(d_out, axis=reduction_axes, keepdims=True)
    d_gamma = arr_mod.sum(d_out * x_hat, axis=reduction_axes, keepdims=True)

    return _ensure_arr((g * inv_std / m) * (m * d_out - d_beta - x_hat * d_gamma))


@global_eager_registry.register("BatchNormGammaGrad")
def _np_batch_norm_gamma_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of BatchNorm with respect to scale (gamma).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Keyword arguments (eps).

    Returns:
        EagerTensorProtocol: Gamma gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    reduction_axes = tuple(i for i in range(x.ndim) if i != 1) if x.ndim > 1 else (0,)
    mean = arr_mod.mean(x, axis=reduction_axes, keepdims=True)
    var = arr_mod.var(x, axis=reduction_axes, keepdims=True)
    x_hat = (x - mean) / arr_mod.sqrt(var + eps)
    return _ensure_arr(arr_mod.sum(d_out * x_hat, axis=reduction_axes).flatten())


@global_eager_registry.register("BatchNormBetaGrad")
def _np_batch_norm_beta_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of BatchNorm with respect to shift (beta).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out].
        **kwargs (BackendArg): Optional keyword arguments.

    Returns:
        EagerTensorProtocol: Beta gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    reduction_axes = tuple(i for i in range(d_out.ndim) if i != 1) if d_out.ndim > 1 else (0,)
    return _ensure_arr(arr_mod.sum(d_out, axis=reduction_axes).flatten())


@global_eager_registry.register("LayerNorm")
def _np_layer_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate LayerNorm forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional arguments [x, gamma, beta].
        **kwargs (BackendArg): Keyword arguments (eps, axis).

    Returns:
        EagerTensorProtocol: Layer-normalized output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    last_dim = x.shape[-1]
    gamma = arr_mod.asarray(args[1]) if len(args) > 1 and args[1] is not None else arr_mod.ones(last_dim, dtype=x.dtype)
    beta = arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.zeros(last_dim, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    mean = adapt_backend_module(arr_mod).require_symbol("mean")(x, axis=axis, keepdims=True)
    var = adapt_backend_module(arr_mod).require_symbol("var")(x, axis=axis, keepdims=True)
    x_hat = (x - mean) / arr_mod.sqrt(var + eps)
    return _ensure_arr(gamma * x_hat + beta)


@global_eager_registry.register("LayerNormInputGrad")
def _np_layer_norm_input_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact analytical gradient of LayerNorm with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x, gamma].
        **kwargs (BackendArg): Keyword arguments (eps, axis).

    Returns:
        EagerTensorProtocol: Input gradient matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    gamma = (
        arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.ones(x.shape[-1], dtype=x.dtype)
    )
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    d = float(x.shape[int(coerce_int(axis))])
    mean = adapt_backend_module(arr_mod).require_symbol("mean")(x, axis=axis, keepdims=True)
    var = adapt_backend_module(arr_mod).require_symbol("var")(x, axis=axis, keepdims=True)
    inv_std = 1.0 / arr_mod.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    d_y = d_out * gamma
    term1 = d * d_y
    term2 = adapt_backend_module(arr_mod).require_symbol("sum")(d_y, axis=axis, keepdims=True)
    term3 = x_hat * adapt_backend_module(arr_mod).require_symbol("sum")(d_y * x_hat, axis=axis, keepdims=True)

    return _ensure_arr((inv_std / d) * (term1 - term2 - term3))


@global_eager_registry.register("LayerNormGammaGrad")
def _np_layer_norm_gamma_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of LayerNorm with respect to gamma.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Keyword arguments (eps, axis).

    Returns:
        EagerTensorProtocol: Gamma gradient.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))
    axis = kwargs.get("axis", -1)

    mean = adapt_backend_module(arr_mod).require_symbol("mean")(x, axis=axis, keepdims=True)
    var = adapt_backend_module(arr_mod).require_symbol("var")(x, axis=axis, keepdims=True)
    x_hat = (x - mean) / arr_mod.sqrt(var + eps)
    batch_axes = tuple(
        i
        for i in range(x.ndim)
        if i != (x.ndim + int(coerce_int(axis)) if int(coerce_int(axis)) < 0 else int(coerce_int(axis)))
    )
    return _ensure_arr(arr_mod.sum(d_out * x_hat, axis=batch_axes))


@global_eager_registry.register("LayerNormBetaGrad")
def _np_layer_norm_beta_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of LayerNorm with respect to beta.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out].
        **kwargs (BackendArg): Keyword arguments (axis).

    Returns:
        EagerTensorProtocol: Beta gradient.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    axis = kwargs.get("axis", -1)
    batch_axes = tuple(
        i
        for i in range(d_out.ndim)
        if i != (d_out.ndim + int(coerce_int(axis)) if int(coerce_int(axis)) < 0 else int(coerce_int(axis)))
    )
    return _ensure_arr(arr_mod.sum(d_out, axis=batch_axes))


@global_eager_registry.register("RMSNorm")
def _np_rms_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate RMSNorm forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional arguments [x, gamma].
        **kwargs (BackendArg): Keyword arguments (eps).

    Returns:
        EagerTensorProtocol: Normalized tensor.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    last_dim = x.shape[-1]
    gamma = arr_mod.asarray(args[1]) if len(args) > 1 and args[1] is not None else arr_mod.ones(last_dim, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    rms = arr_mod.sqrt(arr_mod.mean(arr_mod.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    return _ensure_arr(gamma * x_hat)


@global_eager_registry.register("RMSNormInputGrad")
def _np_rms_norm_input_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact analytical gradient of RMSNorm with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x, gamma].
        **kwargs (BackendArg): Keyword arguments (eps).

    Returns:
        EagerTensorProtocol: Input gradient matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    gamma = (
        arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.ones(x.shape[-1], dtype=x.dtype)
    )
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    d = float(x.shape[-1])
    rms = arr_mod.sqrt(arr_mod.mean(arr_mod.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    d_y = d_out * gamma

    term = x_hat * arr_mod.sum(d_y * x_hat, axis=-1, keepdims=True)
    return _ensure_arr((1.0 / rms) * (d_y - (1.0 / d) * term))


@global_eager_registry.register("RMSNormGammaGrad")
def _np_rms_norm_gamma_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of RMSNorm with respect to gamma.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Keyword arguments (eps).

    Returns:
        EagerTensorProtocol: Gamma gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    rms = arr_mod.sqrt(arr_mod.mean(arr_mod.square(x), axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    batch_axes = tuple(range(x.ndim - 1))
    return _ensure_arr(arr_mod.sum(d_out * x_hat, axis=batch_axes))


@global_eager_registry.register("GroupNorm")
def _np_group_norm(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate GroupNorm forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [x, gamma, beta].
        **kwargs (BackendArg): Keyword arguments (num_groups, groups, eps).

    Returns:
        EagerTensorProtocol: Group-normalized output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(coerce_int(kwargs.get("num_groups", kwargs.get("groups", 1))))
    gamma = arr_mod.asarray(args[1]) if len(args) > 1 and args[1] is not None else arr_mod.ones(c, dtype=x.dtype)
    beta = arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.zeros(c, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    mean = arr_mod.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = arr_mod.var(x_grouped, axis=norm_axes, keepdims=True)
    x_hat = (x_grouped - mean) / arr_mod.sqrt(var + eps)
    x_hat_flat = x_hat.reshape(x.shape)

    b_shape = [1] * x.ndim
    b_shape[1] = c
    return _ensure_arr(gamma.reshape(b_shape) * x_hat_flat + beta.reshape(b_shape))


@global_eager_registry.register("GroupNormInputGrad")
def _np_group_norm_input_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of GroupNorm with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x, gamma].
        **kwargs (BackendArg): Keyword arguments (num_groups, groups, eps).

    Returns:
        EagerTensorProtocol: Input gradient matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(coerce_int(kwargs.get("num_groups", kwargs.get("groups", 1))))
    gamma = arr_mod.asarray(args[2]) if len(args) > 2 and args[2] is not None else arr_mod.ones(c, dtype=x.dtype)
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    b_shape = [1] * x.ndim
    b_shape[1] = c
    d_out_scaled = d_out * gamma.reshape(b_shape)
    d_out_grouped = d_out_scaled.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    m = float(arr_mod.prod([x_grouped.shape[i] for i in norm_axes]))
    mean = arr_mod.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = arr_mod.var(x_grouped, axis=norm_axes, keepdims=True)
    inv_std = 1.0 / arr_mod.sqrt(var + eps)
    x_hat = (x_grouped - mean) * inv_std

    d_beta = arr_mod.sum(d_out_grouped, axis=norm_axes, keepdims=True)
    d_gamma = arr_mod.sum(d_out_grouped * x_hat, axis=norm_axes, keepdims=True)

    dx_grouped = (inv_std / m) * (m * d_out_grouped - d_beta - x_hat * d_gamma)
    return _ensure_arr(dx_grouped.reshape(x.shape))


@global_eager_registry.register("GroupNormGammaGrad")
def _np_group_norm_gamma_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of GroupNorm with respect to gamma.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Keyword arguments (num_groups, groups, eps).

    Returns:
        EagerTensorProtocol: Gamma gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    n, c = x.shape[0], x.shape[1]
    num_groups = int(coerce_int(kwargs.get("num_groups", kwargs.get("groups", 1))))
    eps = float(coerce_float(kwargs.get("eps", kwargs.get("epsilon", 1e-5))))

    c_per_group = c // num_groups
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(n, num_groups, c_per_group, *spatial_shape)

    norm_axes = tuple(range(2, x_grouped.ndim))
    mean = arr_mod.mean(x_grouped, axis=norm_axes, keepdims=True)
    var = arr_mod.var(x_grouped, axis=norm_axes, keepdims=True)
    x_hat = (x_grouped - mean) / arr_mod.sqrt(var + eps)
    x_hat_flat = x_hat.reshape(x.shape)

    axes_to_sum = (0,) + tuple(range(2, x.ndim))
    return _ensure_arr(arr_mod.sum(d_out * x_hat_flat, axis=axes_to_sum))


@global_eager_registry.register("GroupNormBetaGrad")
def _np_group_norm_beta_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of GroupNorm with respect to beta.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out].
        **kwargs (BackendArg): Optional keyword arguments.

    Returns:
        EagerTensorProtocol: Beta gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    axes_to_sum = (0,) + tuple(range(2, d_out.ndim))
    return _ensure_arr(arr_mod.sum(d_out, axis=axes_to_sum))


@global_eager_registry.register("AvgPool2D")
def _np_avg_pool2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate AvgPool2D forward in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional args [x].
        **kwargs (BackendArg): Kernel and stride specs.

    Returns:
        EagerTensorProtocol: Average-pooled output tensor.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_dims = (k, k) if isinstance(k, int) else coerce_shape(k)
    k_h, k_w = k_dims[0], k_dims[1]

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_dims = (s, s) if isinstance(s, int) else coerce_shape(s)
    s_h, s_w = s_dims[0], s_dims[1]

    n, c, h, w = x.shape
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = arr_mod.zeros((n, c, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            out[:, :, i, j] = adapt_backend_module(arr_mod).require_symbol("mean")(
                x[:, :, h_start:h_end, w_start:w_end], axis=(-2, -1)
            )
    return out


@global_eager_registry.register("AvgPool2DGrad")
def _np_avg_pool2d_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of AvgPool2D with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Kernel and stride specs.

    Returns:
        EagerTensorProtocol: Input gradient matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_dims = (k, k) if isinstance(k, int) else coerce_shape(k)
    k_h, k_w = k_dims[0], k_dims[1]

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_dims = (s, s) if isinstance(s, int) else coerce_shape(s)
    s_h, s_w = s_dims[0], s_dims[1]

    dx = arr_mod.zeros_like(x)
    out_h, out_w = d_out.shape[2], d_out.shape[3]
    scale = 1.0 / float(k_h * k_w)

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            dx[:, :, h_start:h_end, w_start:w_end] += d_out[:, :, i : i + 1, j : j + 1] * scale
    return dx


@global_eager_registry.register("MaxPool2DWithArgmaxGrad")
def _np_max_pool2d_with_argmax_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Evaluate exact gradient of MaxPool2DWithArgmax with respect to input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): [d_out, x].
        **kwargs (BackendArg): Kernel and stride specs.

    Returns:
        EagerTensorProtocol: Input gradient matching x shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out = arr_mod.asarray(args[0])
    x = arr_mod.asarray(args[1])
    k = kwargs.get("kernel_size", kwargs.get("pool_size", (2, 2)))
    k_dims = (k, k) if isinstance(k, int) else coerce_shape(k)
    k_h, k_w = k_dims[0], k_dims[1]

    s = kwargs.get("stride", kwargs.get("strides", (k_h, k_w)))
    s_dims = (s, s) if isinstance(s, int) else coerce_shape(s)
    s_h, s_w = s_dims[0], s_dims[1]

    dx = arr_mod.zeros_like(x)
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    for i in range(out_h):
        for j in range(out_w):
            h_start, h_end = i * s_h, i * s_h + k_h
            w_start, w_end = j * s_w, j * s_w + k_w
            window = x[:, :, h_start:h_end, w_start:w_end]
            max_val = adapt_backend_module(arr_mod).require_symbol("max")(window, axis=(-2, -1), keepdims=True)
            mask = arr_mod.equal(window, max_val).astype(x.dtype)
            mask_sum = arr_mod.sum(mask, axis=(-2, -1), keepdims=True)
            mask = mask / arr_mod.maximum(mask_sum, 1.0)
            dx[:, :, h_start:h_end, w_start:w_end] += d_out[:, :, i : i + 1, j : j + 1] * mask
    return dx


def _gelu(
    x: BackendArg,
    *args: float | int | None,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Evaluate Gaussian Error Linear Unit (GELU) activation.

    Args:
        x (BackendArg): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: GELU activated tensor.
    """
    arr_mod = _resolve_eager_mod(None)
    del args, kwargs
    erf_vec = arr_mod.vectorize(math.erf)
    x_f = arr_mod.asarray(x, dtype=float)
    return _ensure_arr(0.5 * x_f * (1.0 + erf_vec(x_f / arr_mod.sqrt(2.0))))


@global_eager_registry.register("Relu")
def _np_relu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: float | int | None,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Evaluate Rectified Linear Unit (ReLU) activation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: ReLU activated tensor.
    """
    del args, kwargs
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("maximum")(x, 0.0))


@global_eager_registry.register("AlphaDropout")
def _np_alpha_dropout(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | bool | int | tuple[int, ...] | None,
) -> EagerTensorProtocol:
    """Evaluate Alpha Dropout for self-normalizing neural networks.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        **kwargs (Union[float, bool, int, tuple[int, ...], None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Output tensor with alpha dropout applied.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    del backend_module
    x_arr = _ensure_arr(x)
    rate_val = kwargs.get("rate", 0.5)
    rate = float(coerce_float(rate_val)) if rate_val is not None else 0.5
    training = bool(kwargs.get("training", False))
    if not training or rate == 0.0:
        return x_arr
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    alpha_p = -alpha * scale
    seed_val = kwargs.get("seed", None)
    rng = arr_mod.random.default_rng(int(coerce_int(seed_val)) if seed_val is not None else None)
    noise_shape_val = kwargs.get("noise_shape", None)
    noise_shape = noise_shape_val if isinstance(noise_shape_val, tuple) else x_arr.shape
    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)
    a = 1.0 / arr_mod.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)
    b = -a * alpha_p * rate
    return _ensure_arr(a * (arr_mod.multiply(x_arr, mask) + alpha_p * (1.0 - mask)) + b)


@global_eager_registry.register("ActivityRegularization")
def _np_activity_regularization(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | None,
) -> EagerTensorProtocol:
    """Evaluate activity regularization passthrough.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        **kwargs (Union[float, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Unmodified input tensor.
    """
    del backend_module, kwargs
    return _ensure_arr(x)


@global_eager_registry.register("Dropout")
def _np_dropout(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    rate: float = 0.5,
    **kwargs: float | bool | int | tuple[int, ...] | None,
) -> EagerTensorProtocol:
    """Evaluate inverted dropout on tensor.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        rate (float): Dropout probability rate.
        **kwargs (Union[float, bool, int, tuple[int, ...], None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Dropout applied tensor.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    del backend_module
    x_arr = _ensure_arr(x)
    rate_val = kwargs.get("rate", rate)
    rate_f = float(coerce_float(rate_val)) if rate_val is not None else rate
    training = bool(kwargs.get("training", False))
    if not training or rate_f == 0.0:
        return x_arr
    seed_val = kwargs.get("seed", None)
    rng = arr_mod.random.default_rng(int(coerce_int(seed_val)) if seed_val is not None else None)
    noise_shape_val = kwargs.get("noise_shape", None)
    noise_shape = noise_shape_val if isinstance(noise_shape_val, tuple) else x_arr.shape
    mask = rng.binomial(1, 1.0 - rate_f, size=noise_shape)
    return _ensure_arr(arr_mod.divide(arr_mod.multiply(x_arr, mask), 1.0 - rate_f))


@global_eager_registry.register("TimeDistributed")
def _np_time_distributed(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: str | float | int | None,
) -> EagerTensorProtocol:
    """Apply an operation over time steps of a sequential tensor.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input sequence tensor.
        **kwargs (Union[str, float, int, None]): Keyword arguments including wrapped_op_name.

    Returns:
        EagerTensorProtocol: Time distributed transformed tensor.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    del backend_module
    x_arr = _ensure_arr(x)
    wrapped_op_name = str(kwargs.pop("wrapped_op_name"))
    shape = x_arr.shape
    if len(shape) < MIN_CONV_RANK:
        return arr_mod.asarray(get_active_backend().execute_op(wrapped_op_name, x_arr, **kwargs))
    flat_x = arr_mod.reshape(x_arr, (shape[0] * shape[1], *shape[2:]))
    out = get_active_backend().execute_op(wrapped_op_name, flat_x, **kwargs)
    out_shape = (shape[0], shape[1], *coerce_shape(adapt_backend_module(out).require_symbol("shape"))[1:])
    return _ensure_arr(adapt_backend_module(arr_mod).require_symbol("reshape")(out, out_shape))


@global_eager_registry.register("Rope")
def _np_rope(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: int | float | None,
) -> EagerTensorProtocol:
    """Apply Rotary Positional Encoding (RoPE) to embeddings.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        **kwargs (Union[int, float, None]): Keyword arguments (dim, axis, offset, base).

    Returns:
        EagerTensorProtocol: RoPE encoded embeddings.
    """
    x_np = adapt_backend_module(backend_module).require_symbol("asarray")(x)
    dim_val = kwargs.get("axis", kwargs.get("dim", x_np.shape[-1]))
    dim_int = int(dim_val) if dim_val is not None else x_np.shape[-1]
    half_dim = dim_int // 2
    offset_val = kwargs.get("offset", 0)
    offset = int(offset_val) if offset_val is not None else 0
    base_val = kwargs.get("base", 10000.0)
    base = float(base_val) if base_val is not None else 10000.0

    position = adapt_backend_module(backend_module).require_symbol("arange")(
        offset, offset + x_np.shape[-2], dtype=x_np.dtype
    )
    freqs = adapt_backend_module(backend_module).require_symbol("exp")(
        -adapt_backend_module(backend_module).require_symbol("arange")(0, half_dim, dtype=x_np.dtype)
        * (adapt_backend_module(backend_module).require_symbol("log")(base) / half_dim)
    )
    angles = position[:, None] * freqs[None, :]

    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("concatenate")(
            [
                x_np[..., :half_dim] * adapt_backend_module(backend_module).require_symbol("cos")(angles)
                - x_np[..., half_dim:] * adapt_backend_module(backend_module).require_symbol("sin")(angles),
                x_np[..., :half_dim] * adapt_backend_module(backend_module).require_symbol("sin")(angles)
                + x_np[..., half_dim:] * adapt_backend_module(backend_module).require_symbol("cos")(angles),
            ],
            axis=-1,
        )
    )


@global_eager_registry.register("Rrelu")
def _np_rrelu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: float | int | None,
    **kwargs: float | bool | None,
) -> EagerTensorProtocol:
    """Evaluate Randomized Leaky Rectified Linear Unit (RReLU).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, bool, None]): Keyword arguments (lower, upper, training).

    Returns:
        EagerTensorProtocol: RReLU activated tensor.
    """
    del args
    lower_val = kwargs.get("lower", 1.0 / 8.0)
    lower = float(lower_val) if lower_val is not None else 1.0 / 8.0
    upper_val = kwargs.get("upper", 1.0 / 3.0)
    upper = float(upper_val) if upper_val is not None else 1.0 / 3.0
    training = bool(kwargs.get("training", False))

    x_data = adapt_backend_module(backend_module).require_symbol("asarray")(
        adapt_backend_module(x).get_symbol("data", x)
    )
    if not training:
        alpha = (lower + upper) / 2.0
        return _ensure_arr(
            adapt_backend_module(backend_module).require_symbol("where")(x_data >= 0, x_data, x_data * alpha)
        )

    alpha_arr = adapt_backend_module(backend_module).require_symbol("random").uniform(lower, upper, size=x_data.shape)
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("where")(x_data >= 0, x_data, x_data * alpha_arr)
    )


@global_eager_registry.register("Convolve")
def _np_convolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed result.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("convolve")(*args, **kwargs))
