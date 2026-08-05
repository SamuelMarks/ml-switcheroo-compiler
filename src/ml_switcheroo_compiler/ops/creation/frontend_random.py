"""Constants & Creation Operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .frontend_utils import _emit_creation_node


def rand(
    *size: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random numbers from a uniform distribution.

    Args:
        *size: Additional arguments.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Rand", *shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Rand", shape, dtype)


def randn(
    *size: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random numbers from a standard normal distribution.

    Args:
        *size: Additional arguments.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Randn", *shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Randn", shape, dtype)


def randint(
    low: int,
    high: int,
    size: Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random integers from [low, high).

    Args:
        low (int): The low parameter for the operation.
        high (int): The high parameter for the operation.
        size (Sequence[int]): The size parameter for the operation.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_int_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Randint", low, high, size=shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Randint", shape, dtype, {"low": low, "high": high})


def manual_seed(seed: int) -> int:
    """Set the seed for generating random numbers.

    Args:
        seed (int): The random seed.

    Returns:
        int: The computed result.
    """
    if config.eager_mode:
        get_active_backend().execute_op("Seed", seed)
        return seed
    _emit_creation_node("ManualSeed", (), config.default_int_dtype, {"seed": seed})
    return seed
