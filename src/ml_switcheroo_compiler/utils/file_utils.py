# ruff: noqa: E402, E501, F403
"""File utilities."""

from __future__ import annotations

import os


def exists(path: str) -> bool:
    """Return True if path exists.

    Args:
        path (str): The path parameter.

    Returns:
        bool: Result.
    """
    return os.path.exists(path)
