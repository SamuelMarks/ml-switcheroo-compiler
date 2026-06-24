"""File utilities."""

import os  # pragma: no cover


def exists(path: str) -> bool:  # pragma: no cover
    """Returns True if path exists."""
    return os.path.exists(path)  # pragma: no cover
