# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Debugging and diagnostic hooks."""


def enable_dump_debug_info(dump_root: str, tensor_debug_mode: str = "NO_TENSOR", circular_buffer_size: int = -1) -> None:
    """Enable dumping of debug information during execution.

    Args:
        dump_root (str): The dump_root parameter.
        tensor_debug_mode (str): The tensor_debug_mode parameter.
        circular_buffer_size (int): The circular_buffer_size parameter.
    """
    import os

    os.makedirs(dump_root, exist_ok=True)
