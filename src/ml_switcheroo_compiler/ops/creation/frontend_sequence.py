from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Constants & Creation Operations."""


import math

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .frontend_utils import _emit_creation_node


def arange(
    start: float = 0,
    stop: float | None = None,
    step: float = 1,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Any:
    """Return a 1-D tensor of size with values from the interval `[start, stop)`.

    Args:
        start (float): The start parameter.
        stop (object): The stop parameter.
        step (float): The step parameter.
        dtype (object): The dtype parameter.
        device (object): The device parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    if stop is None:
        stop = start
        start = 0

    size = max(math.ceil((stop - start) / step), 0)
    shape = (size,)

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Arange",
            start,
            stop,
            step,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "Arange",
        shape,
        dtype,
        {"start": start, "stop": stop, "step": step},
    )


def linspace(
    start: float,
    stop: float,
    steps: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Any:
    """Create a 1D tensor evenly spaced from `start` to `stop`.

    Args:
        start (float): The start parameter.
        stop (float): The stop parameter.
        steps (int): The steps parameter.
        dtype (object): The dtype parameter.
        device (object): The device parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (steps,)

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Linspace",
            start,
            stop,
            steps,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "LinSpace",
        shape,
        dtype,
        {"start": start, "stop": stop, "steps": steps},
    )
