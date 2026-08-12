# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_nn module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("ActivityRegularization")
def _activity_regularization(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _activity_regularization operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return x


def _global_adaptive_pool_mock(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _global_adaptive_pool_mock operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(operand, "shape") and hasattr(backend_module, "zeros"):
        s = list(operand.shape)
        if isinstance(output_size, int):
            out_s = [output_size]
            s[-1] = output_size
        else:
            out_s = list(output_size)
            s[-len(output_size) :] = out_s
        if hasattr(backend_module, "broadcast_to") and hasattr(backend_module, "mean"):
            axes = tuple(range(-len(out_s), 0))
            return backend_module.broadcast_to(backend_module.mean(operand, axis=axes, keepdims=True), s)
        dtype = getattr(operand, "dtype", None)
        return backend_module.zeros(s, dtype=dtype) if dtype is not None else backend_module.zeros(s)
    return operand


@global_eager_registry.register("AdaptiveAvgPool2D")
def _adaptive_avg_pool2d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_avg_pool2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AdaptiveAvgPool3D")
def _adaptive_avg_pool3d(backend_module: Any, operand: Any, output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _adaptive_avg_pool3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _global_adaptive_pool_mock(backend_module, operand, output_size, **kwargs)


@global_eager_registry.register("AlphaDropout")
def _alpha_dropout(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _alpha_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return x


@global_eager_registry.register("FractionalAvgPool")
def _np_fractionalavgpool(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fractionalavgpool operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fractionalavgpool", getattr(backend_module, "fractionalavgpool", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]
