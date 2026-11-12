"""Constants & Creation Operations."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .frontend_utils import _dtype_to_str, _emit_creation_node


def arange(
    start: float = 0,
    stop: float | None = None,
    step: float = 1,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Return a 1-D tensor of size with values from the interval `[start, stop)`.

    Args:
        start (float): Input `start` argument (start).
        stop (float | None): Input `stop` argument (stop).
        step (float): Input `step` argument (step).
        dtype (DType | None): Input `dtype` argument (dtype).
        device (Device | None): Input `device` argument (device).

    Returns:
        Tensor: Computed `Tensor` output.
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
            dtype=_dtype_to_str(dtype),
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
) -> TensorOrScalar:
    """Create a 1D tensor evenly spaced from `start` to `stop`.

    Args:
        start (float): Input `start` argument (start).
        stop (float): Input `stop` argument (stop).
        steps (int): Input `steps` argument (steps).
        dtype (DType | None): Input `dtype` argument (dtype).
        device (Device | None): Input `device` argument (device).

    Returns:
        Tensor: Computed `Tensor` output.
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
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "LinSpace",
        shape,
        dtype,
        {"start": start, "stop": stop, "steps": steps},
    )
