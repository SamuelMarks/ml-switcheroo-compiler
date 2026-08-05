"""File utilities."""

import os


def exists(path: str) -> bool:
    """Return True if path exists.

    Args:
        path (str): The path parameter.

    Returns:
        bool: Result.
    """
    return os.path.exists(path)
