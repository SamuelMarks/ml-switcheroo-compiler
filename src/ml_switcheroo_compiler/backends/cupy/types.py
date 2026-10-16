"""Backend utilities."""

from typing import Callable, Optional, cast

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

try:
    import cupy as cp
except ImportError:
    cp = None


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        object: The zeros tensor.
    """
    return generic_zeros(cp, shape)


def array(cls: type, data: EagerValue, dtype: Optional[EagerValue] = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        object: The array tensor.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_array)(cp, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data: The input data.

    Returns:
        object: The array tensor.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_asarray)(cp, data))


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return cast(float, cast("Callable[..., EagerTensorProtocol]", generic_item)(cp, data))
