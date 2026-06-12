"""Provides utility functions for inspecting the call stack and retrieving source code."""

from __future__ import annotations

import inspect


def get_source_ast_ref(back_frames: int = 1) -> str | None:
    """Retrieves the file path and line number of a caller frame at a specified depth in.

    the call stack

    This function traverses the call stack backward by the specified number of
    frames
    and returns a formatted string containing the filename and line number of the
    resulting frame. It is useful for dynamically linking log messages or errors
    back to their source origin

    Args:
    back_frames (int): The number of frames to traverse backward from the caller's
        frame to locate the target caller. Defaults to 1

    Returns:
    str | None: A formatted string in the format "filepath:lineno" representing
    the source location, or None if the frame cannot be retrieved or an error
    occurs
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
    except Exception:
        return None
    finally:
        del frame
