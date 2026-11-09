"""Debugging and diagnostic hooks."""

from __future__ import annotations

import os

_DEBUG_DUMP_CONFIG: dict[str, str | int | bool] = {
    "enabled": False,
    "dump_root": "",
    "tensor_debug_mode": "NO_TENSOR",
    "circular_buffer_size": -1,
}


def get_debug_dump_config() -> dict[str, str | int | bool]:
    """Return a copy of the active debug dump configuration.

    Returns:
        dict[str, str | int | bool]: Active debug dump configuration state.
    """
    return dict(_DEBUG_DUMP_CONFIG)


def enable_dump_debug_info(dump_root: str, tensor_debug_mode: str = "NO_TENSOR", circular_buffer_size: int = -1) -> None:
    """Enable dumping of debug information during execution.

    Args:
        dump_root (str): Destination root directory for debug dumps.
        tensor_debug_mode (str): Tensor debug capture mode (e.g. "NO_TENSOR", "FULL_TENSOR").
        circular_buffer_size (int): Maximum circular buffer size (-1 for unbounded).
    """
    os.makedirs(dump_root, exist_ok=True)
    _DEBUG_DUMP_CONFIG["enabled"] = True
    _DEBUG_DUMP_CONFIG["dump_root"] = dump_root
    _DEBUG_DUMP_CONFIG["tensor_debug_mode"] = tensor_debug_mode
    _DEBUG_DUMP_CONFIG["circular_buffer_size"] = circular_buffer_size
