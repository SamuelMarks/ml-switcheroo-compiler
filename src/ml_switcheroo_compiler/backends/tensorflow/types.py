"""Backend utilities."""

from typing import Callable, cast

import tensorflow as tf

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): The class.
    shape (tuple): The shape parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return generic_zeros(tf, shape)


def array(cls: type, data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.
        dtype (EagerValue): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_array)(tf, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_asarray)(tf, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
        float: Result.
    """
    return cast(float, cast("Callable[..., EagerTensorProtocol]", generic_item)(tf, data))
