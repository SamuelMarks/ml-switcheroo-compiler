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
