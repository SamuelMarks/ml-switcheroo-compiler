"""Backend utilities."""


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Execute zeros.

    Args:
        cls (Any): The cls parameter for the operation.
        shape (Any): Argument shape.

    Returns:
    Any: The result.
    """
    import torch

    return torch.zeros(shape)


def array(cls: type, data: object) -> object:
    """Execute array.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    import torch

    return torch.tensor(data)


def asarray(cls: type, data: object) -> object:
    """Execute asarray.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    import torch

    return torch.as_tensor(data)


def item(cls: type, data: object) -> float:
    """Execute item.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    import torch

    return torch.as_tensor(data).item()
