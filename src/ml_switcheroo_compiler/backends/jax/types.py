"""Backend utilities."""

from typing import Callable, Optional, cast

import jax.numpy as jnp

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager.types_utils import ZerosModule
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Evaluate zeros operation.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            object: Result.
    """
    return generic_zeros(cast(ZerosModule, jnp), shape)


def array(cls: type, data: EagerValue, dtype: Optional[EagerValue] = None) -> EagerValue:
    """Evaluate array operation.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_array)(jnp, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Evaluate asarray operation.

    Args:
        cls: Class.
        data: Data.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_asarray)(jnp, data))


def item(cls: type, data: EagerValue) -> float:
    """Evaluate item operation.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return cast(float, cast("Callable[..., EagerTensorProtocol]", generic_item)(jnp, data))
