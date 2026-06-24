"""Dummy distributed primitives for Numpy backend."""


def _dummy_all_gather(tensor: object, axis: int, mesh: object) -> object:
    """Function docstring.

    Args:
        tensor: Arg.
        axis: Arg.
        mesh: Arg.
    """
    # In a real environment, this would gather tensors across devices.
    # Here we just return the tensor since we are simulating.
    return tensor


def _dummy_reduce_scatter(tensor: object, op: str, axis: int, mesh: object) -> object:
    """Function docstring.

    Args:
        tensor: Arg.
        op: Arg.
        axis: Arg.
        mesh: Arg.
    """
    # Simulates reduce_scatter.
    return tensor


def _dummy_all_reduce(tensor: object, op: str, mesh: object) -> object:
    """Function docstring.

    Args:
        tensor: Arg.
        op: Arg.
        mesh: Arg.
    """
    # Simulates all_reduce.
    return tensor
