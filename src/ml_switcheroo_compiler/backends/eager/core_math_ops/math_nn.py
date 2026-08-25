# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_nn module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(backend_module: typing.Any, x: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _activity_regularization operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


def _global_adaptive_pool(backend_module: typing.Any, operand: typing.Any, output_size: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _global_adaptive_pool operation rigorously over spatial dimensions.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import math

    if not hasattr(operand, "shape"):
        return operand

    if isinstance(output_size, int):
        out_spatial: typing.Any = [output_size]
    else:
        out_spatial: typing.Any = list(output_size)

    spatial_dims: typing.Any = len(out_spatial)
    in_shape: typing.Any = operand.shape
    if len(in_shape) < spatial_dims:
        return operand
    in_spatial: typing.Any = in_shape[-spatial_dims:]

    if spatial_dims == 1:
        O_w: typing.Any = out_spatial[0]
        I_w: typing.Any = in_spatial[0]
        bins: typing.Any = []
        for j in range(O_w):
            start: typing.Any = math.floor(j * I_w / O_w)
            end: typing.Any = math.ceil((j + 1) * I_w / O_w)
            bins.append(backend_module.mean(operand[..., start:end], axis=-1))
        return backend_module.stack(bins, axis=-1)

    if spatial_dims == 2:
        O_h, O_w = out_spatial
        I_h, I_w = in_spatial
        rows: typing.Any = []
        for i in range(O_h):
            h_start: typing.Any = math.floor(i * I_h / O_h)
            h_end: typing.Any = math.ceil((i + 1) * I_h / O_h)
            cols: typing.Any = []
            for j in range(O_w):
                w_start: typing.Any = math.floor(j * I_w / O_w)
                w_end: typing.Any = math.ceil((j + 1) * I_w / O_w)
                cols.append(backend_module.mean(operand[..., h_start:h_end, w_start:w_end], axis=(-2, -1)))
            rows.append(backend_module.stack(cols, axis=-1))
        return backend_module.stack(rows, axis=-2)

    if spatial_dims == 3:
        O_d, O_h, O_w = out_spatial
        I_d, I_h, I_w = in_spatial
        depths: typing.Any = []
        for k in range(O_d):
            d_start: typing.Any = math.floor(k * I_d / O_d)
            d_end: typing.Any = math.ceil((k + 1) * I_d / O_d)
            rows: typing.Any = []
            for i in range(O_h):
                h_start: typing.Any = math.floor(i * I_h / O_h)
                h_end: typing.Any = math.ceil((i + 1) * I_h / O_h)
                cols: typing.Any = []
                for j in range(O_w):
                    w_start: typing.Any = math.floor(j * I_w / O_w)
                    w_end: typing.Any = math.ceil((j + 1) * I_w / O_w)
                    cols.append(backend_module.mean(operand[..., d_start:d_end, h_start:h_end, w_start:w_end], axis=(-3, -2, -1)))
                rows.append(backend_module.stack(cols, axis=-1))
            depths.append(backend_module.stack(rows, axis=-2))
        return backend_module.stack(depths, axis=-3)

    return operand


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(backend_module: typing.Any, operand: typing.Any, output_size: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _adaptive_avg_pool2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(backend_module: typing.Any, operand: typing.Any, output_size: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _adaptive_avg_pool3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _global_adaptive_pool(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(backend_module: typing.Any, x: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _alpha_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


@global_eager_registry.register("FractionalAvgPool")
def _np_fractionalavgpool(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> object:
    """Evaluate _np_fractionalavgpool operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: typing.Any = getattr(backend_module, "fractionalavgpool", getattr(backend_module, "fractionalavgpool", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]
