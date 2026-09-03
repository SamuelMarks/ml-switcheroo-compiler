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
