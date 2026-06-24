"""Dispatch utilities for the ml-switcheroo compiler."""

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.backends.registry import get_active_backend


def dispatch(module_name: str, func_name: str, *args: object, **kwargs: object) -> object:
    """Dynamically dispatch a function to the active backend.

    Args:
        module_name (str): The name of the backend submodule (e.g., 'lax').
        func_name (str): The name of the function to execute (e.g., 'abs_p').
        *args (object): Positional arguments for the function.
        **kwargs (object): Keyword arguments for the function.

    Returns:
        object: The result of the function execution.

    Raises:
        NotImplementedError: If not supported in the active backend or in tracing mode.
    """
    if core_config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, module_name):  # pragma: no branch
            submodule = getattr(backend.module, module_name)  # pragma: no cover
            if hasattr(submodule, func_name):  # pragma: no cover
                return getattr(submodule, func_name)(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(
            f"{func_name} is not supported in the active backend."
        )  # pragma: no cover
    raise NotImplementedError(f"{func_name} is not fully supported in tracing mode.")
