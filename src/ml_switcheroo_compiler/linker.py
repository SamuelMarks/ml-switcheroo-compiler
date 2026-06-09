"""Module docstring."""

import inspect
from typing import Optional


def get_source_ast_ref(back_frames: int = 1) -> Optional[str]:
    """Dynamically link to source AST references using inspect.currentframe().

    Args:
        back_frames: The number of frames to go back to find the user's caller.

    Returns:
        Optional[str]: A formatted string "filepath:lineno" or None if unavailable.
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
