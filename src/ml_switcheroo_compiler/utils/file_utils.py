# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module file_utils.py."""

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
