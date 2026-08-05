# ruff: noqa: E501
"""Backend utilities."""

try:
    import dask.array as da
except ImportError:
    da = None


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        object: The zeros tensor.
    """
    return da.zeros(shape, chunks="auto")


def array(cls: type, data: object, dtype: object = None) -> object:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data (object): The input data.
        dtype (object): The target dtype.

    Returns:
        object: The array tensor.
    """
    if dtype is not None:
        return da.array(data, dtype=getattr(dtype, "value", dtype))
    return da.array(data)


def asarray(cls: type, data: object) -> object:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data (object): The input data.

    Returns:
        object: The array tensor.
    """
    return da.asarray(data)


def item(cls: type, data: object) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data (object): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return float(da.asarray(data).compute().item())
