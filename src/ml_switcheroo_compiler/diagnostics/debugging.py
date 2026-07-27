"""Debugging and diagnostic hooks."""


def enable_dump_debug_info(dump_root: str, tensor_debug_mode: str = "NO_TENSOR", circular_buffer_size: int = -1) -> None:
    """Enable dumping of debug information during execution."""
    import os

    os.makedirs(dump_root, exist_ok=True)
