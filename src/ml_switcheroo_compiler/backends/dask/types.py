"""Backend utilities."""

from types import ModuleType
from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue

try:
    import dask.array as da
except ImportError:
    da = cast(ModuleType, None)


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        EagerValue: The zeros tensor.
    """
    return cast(EagerValue, getattr(da, "zeros")(shape, chunks="auto"))


def array(cls: type, data: EagerValue, dtype: Optional[EagerValue] = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        EagerValue: The array tensor.
    """
    if dtype is not None:
        return cast(EagerValue, getattr(da, "array")(data, dtype=getattr(dtype, "value", dtype)))
    return cast(EagerValue, getattr(da, "array")(data))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data: The input data.

    Returns:
        EagerValue: The array tensor.
    """
    return cast(EagerValue, getattr(da, "asarray")(data))


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return float(getattr(da, "asarray")(data).compute().item())
