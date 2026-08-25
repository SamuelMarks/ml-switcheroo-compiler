"""Module linker.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide utility functions for inspecting the call stack and retrieving source code."""


import inspect


def get_source_ast_ref(back_frames: int = 1) -> str | None:
    """Retrieve the file path and line number of a caller frame at a specified depth in.

    Args:
        back_frames (int): The back_frames parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    try:
        frame: object = inspect.currentframe()
        if frame is None:
            return None

        # Traverse back the specified number of frames
        for _ in range(back_frames + 1):
            if frame.f_back:
                frame: object = frame.f_back
            else:
                break

        info: object = inspect.getframeinfo(frame)
        return f"{info.filename}:{info.lineno}"
    except (ValueError, TypeError, AttributeError):
        return None
    finally:
        del frame
