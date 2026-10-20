"""Provide utility functions for inspecting the call stack and retrieving source code."""

from __future__ import annotations

import inspect
import typing  # noqa: F401


def get_source_ast_ref(back_frames: int = 1) -> str | None:
    """Retrieve the file path and line number of a caller frame at a specified depth in.

    Args:
        back_frames (int): The back_frames parameter.

    Returns:
            Optional[str]: Result.
    """
    try:
        frame = inspect.currentframe()
        if frame is None:
            return None

        # Traverse back the specified number of frames
        for _ in range(back_frames + 1):
            if frame.f_back:
                frame = frame.f_back
            else:
                break

        info = inspect.getframeinfo(frame)
        return f"{info.filename}:{info.lineno}"
    except (ValueError, TypeError, AttributeError):
        return None
    finally:
        del frame
