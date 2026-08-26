# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


@numpy_eager_registry.register("Clz")
def _np_clz(backend_module, x, *args, **kwargs):
    """Count the number of leading zero bits in the integer representation of the input.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        TypeError: An exception.
    """
    x_arr = np.asarray(x)
    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError("Clz requires integer inputs.")
    bit_width = x_arr.itemsize * 8

    @np.vectorize
    def _clz_scalar(val):
        """Evaluate _clz_scalar operation.

        Args:
        val (object): The val parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        val = int(val)
        if val < 0:
            val = val & (1 << bit_width) - 1
        return bit_width - val.bit_length()

    res = _clz_scalar(x_arr)
    return res.astype(x_arr.dtype)


@numpy_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(backend_module, x, new_dtype, *args, **kwargs):
    """Bitcast a tensor from one type to another without changing its underlying memory.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        new_dtype (object): The new_dtype parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    dt = getattr(np, str(new_dtype).split(".")[-1], np.float32)
    return np.asarray(x).view(dt)


@numpy_eager_registry.register("Packbits")
def _np_packbits(backend_module, *args, **kwargs):
    """Pack the elements of a binary-valued array into bits in a uint8 array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return np.packbits(np.asarray(args[0]), **kwargs)


@numpy_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module, *args, **kwargs):
    """Unpack elements of a uint8 array into a binary-valued output array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return np.unpackbits(np.asarray(args[0]), **kwargs)


@numpy_eager_registry.register("BitwiseCount")
def _np_bitwise_count(backend_module, *args, **kwargs):
    """Evaluate _np_bitwise_count operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x = np.asarray(args[0])
    return np.array([bin(n).count("1") for n in x.flat]).reshape(x.shape)
