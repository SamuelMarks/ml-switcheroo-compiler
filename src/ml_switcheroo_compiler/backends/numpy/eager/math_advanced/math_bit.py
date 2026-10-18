"""Math Ops."""

from collections.abc import Sequence  # noqa: F401
from types import ModuleType
from typing import (
    Optional,  # noqa: F401
    cast,
)

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401

from .math_general import _get_np_arg, _get_sc  # noqa: F401


@numpy_eager_registry.register("Clz")
def _np_clz(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Count the number of leading zero bits in the integer representation of the input.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        TypeError: An exception.
    """
    x_arr = np.asarray(x)
    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError("Clz requires integer inputs.")
    bit_width = x_arr.itemsize * 8

    def _clz_scalar(val: EagerValue) -> int:
        """Execute the eager clz scalar operation on the active backend array.

        Args:
        val (EagerValue): The val parameter.

        Returns:
            int: Result.
        """
        val_i: int = int(cast(int, val))
        if val_i < 0:
            val_i = val_i & (1 << int(bit_width)) - 1
        return int(bit_width) - val_i.bit_length()

    res = np.vectorize(_clz_scalar)(x_arr)
    return cast(EagerValue, res.astype(x_arr.dtype))


@numpy_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(backend_module: ModuleType, x: EagerValue, new_dtype: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Bitcast a tensor from one type to another without changing its underlying memory.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        new_dtype (EagerValue): The new_dtype parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    dt = getattr(np, str(new_dtype).split(".")[-1], np.float32)
    return np.asarray(x).view(dt)


@numpy_eager_registry.register("Packbits")
def _np_packbits(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Pack the elements of a binary-valued array into bits in a uint8 array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "packbits")(np.asarray(args[0]), **kwargs))


@numpy_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Unpack elements of a uint8 array into a binary-valued output array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "unpackbits")(np.asarray(args[0]), **kwargs))


@numpy_eager_registry.register("BitwiseCount")
def _np_bitwise_count(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np bitwise count operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x = np.asarray(args[0])
    return np.array([bin(n).count("1") for n in x.flat]).reshape(x.shape)
