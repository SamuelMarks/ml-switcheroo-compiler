"""Ahead-of-Time compilation hooks for frontend integrations."""

from collections.abc import Callable


def compile_function(fn: Callable[..., object], backend: str = "numpy", **kwargs: object) -> Callable[..., object]:
    """Compiles a function functionally, intended for torch.compile integrations.

    Args:
        fn: The function to compile.
        backend: The target execution backend.
        kwargs: Additional compilation options.

    Returns:
        Callable: The compiled function.
    """

    def compiled_wrapper(*args: object, **kw: object) -> object:
        return fn(*args, **kw)

    return compiled_wrapper
