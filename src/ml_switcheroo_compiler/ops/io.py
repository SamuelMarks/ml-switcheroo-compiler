"""I/O and memory operations."""


def load(*args: object, **kwargs: object) -> object:
    """Mock load."""
    raise NotImplementedError("load not implemented")  # pragma: no cover


def save(*args: object, **kwargs: object) -> object:
    """Mock save."""
    raise NotImplementedError("save not implemented")  # pragma: no cover


def save_gguf(*args: object, **kwargs: object) -> object:
    """Mock save_gguf."""
    raise NotImplementedError("save_gguf not implemented")  # pragma: no cover


def save_safetensors(*args: object, **kwargs: object) -> object:
    """Mock save_safetensors."""
    raise NotImplementedError("save_safetensors not implemented")  # pragma: no cover


def savez(*args: object, **kwargs: object) -> object:
    """Mock savez."""
    raise NotImplementedError("savez not implemented")  # pragma: no cover


def savez_compressed(*args: object, **kwargs: object) -> object:
    """Mock savez_compressed."""
    raise NotImplementedError("savez_compressed not implemented")  # pragma: no cover


def set_default_stream(*args: object, **kwargs: object) -> object:
    """Mock set_default_stream."""
    raise NotImplementedError("set_default_stream not implemented")  # pragma: no cover


def set_memory_limit(*args: object, **kwargs: object) -> object:
    """Mock set_memory_limit."""
    raise NotImplementedError("set_memory_limit not implemented")  # pragma: no cover


def set_wired_limit(*args: object, **kwargs: object) -> object:
    """Mock set_wired_limit."""
    raise NotImplementedError("set_wired_limit not implemented")  # pragma: no cover


__all__ = [
    "load",
    "save",
    "save_gguf",
    "save_safetensors",
    "savez",
    "savez_compressed",
    "set_default_stream",
    "set_memory_limit",
    "set_wired_limit",
]
