"""Constants & Creation Operations."""

from __future__ import annotations


from typing import TYPE_CHECKING


from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType


from .frontend_utils import _emit_creation_node


def arange(
    start: float = 0,
    stop: float | None = None,
    step: float = 1,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a 1-D tensor of size with values from the interval `[start, stop)`.

    Args:
        start (Union[float, int]): Argument start
        stop (Optional[Union[float, int]]): Argument stop
        step (Union[float, int]): Argument step
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    if stop is None:
        stop = start
        start = 0
    import math

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
) -> Tensor:
    """Create a 1D tensor evenly spaced from `start` to `stop`.

    Args:
        start (Union[float, int]): Argument start
        stop (Union[float, int]): Argument stop
        steps (int): Argument steps
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
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
