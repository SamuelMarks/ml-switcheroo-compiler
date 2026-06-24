"""Common utility for aliases."""

from typing import Callable

from ml_switcheroo_compiler.core.config import config as core_config


def create_eager_alias(name: str) -> Callable[..., object]:
    """Create an eager execution alias for a backend operation."""

    def alias(*args: object, **kwargs: object) -> object:
        """Function docstring.

        Args:
        args: Arg.
        kwargs: Arg.
        """
        if core_config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            return backend.execute_op(name, *args, **kwargs)
        raise NotImplementedError(f"{name} is not fully supported in tracing mode.")

    alias.__name__ = name
    alias.__doc__ = f"Execute {name}."
    return alias
