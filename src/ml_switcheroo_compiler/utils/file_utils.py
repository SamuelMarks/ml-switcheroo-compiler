"""Internal filesystem path utilities for the compiler."""

from __future__ import annotations

import os


def exists(path: str) -> bool:
    """Check whether a filesystem path exists on disk.

    Args:
        path (str): Target filesystem path to check.

    Returns:
        bool: True if ``path`` exists, False otherwise.
    """
    return os.path.exists(path)
