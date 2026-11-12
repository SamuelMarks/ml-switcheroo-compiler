"""File utilities."""

from __future__ import annotations

import os


def exists(path: str) -> bool:
    """Return True if path exists.

    Args:
        path (str): Input `path` argument (path).

    Returns:
        bool: Computed `bool` output.
    """
    return os.path.exists(path)
