"""File utilities."""

import os


def exists(path: str) -> bool:
    """Returns True if path exists."""
    return os.path.exists(path)
