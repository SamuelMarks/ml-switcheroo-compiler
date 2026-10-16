"""Backend utilities."""

import torch

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Evaluate zeros operation.

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
    return torch.zeros(shape)


def array(cls: type, data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Evaluate array operation.

    Args:
        cls (type): Class.
        data (object): Data.
        dtype (object): Dtype.

    Args:
        cls (type): Class.
        data (object): Data.
        dtype (object): Dtype.

    Args:
        cls (type): The class.
    data (object): The data parameter.
        dtype (object): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if dtype is not None:
        dtype_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
        dt = getattr(torch, dtype_str, None)
        return torch.tensor(data, dtype=dt)
    return torch.tensor(data)


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Evaluate asarray operation.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): The class.
    data (object): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return torch.as_tensor(data)


def item(cls: type, data: EagerValue) -> float:
    """Evaluate item operation.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): The class.
    data (object): The data parameter.

    Returns:
        float: Result.
    """
    return float(torch.as_tensor(data).item())
