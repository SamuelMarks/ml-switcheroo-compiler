"""Dispatch utilities for the ml-switcheroo compiler."""

from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover
from ml_switcheroo_compiler.core.config import config as core_config  # pragma: no cover


# pragma: no cover
# pragma: no cover
def dispatch(module_name: str, func_name: str, *args: object, **kwargs: object) -> object:  # pragma: no cover
    """Dynamically dispatch a function to the active backend.

    Args:  # pragma: no cover
        module_name (str): The name of the backend submodule (e.g., 'lax').  # pragma: no cover
        func_name (str): The name of the function to execute (e.g., 'abs_p').  # pragma: no cover
        *args (object): Positional arguments for the function.  # pragma: no cover
        **kwargs (object): Keyword arguments for the function.  # pragma: no cover
    # pragma: no cover
    Returns:  # pragma: no cover
        object: The result of the function execution.  # pragma: no cover
    # pragma: no cover
    Raises:  # pragma: no cover
        NotImplementedError: If not supported in the active backend or in tracing mode.  # pragma: no cover
    """  # pragma: no cover
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        if hasattr(backend.module, module_name):  # pragma: no branch  # pragma: no cover
            submodule = getattr(backend.module, module_name)  # pragma: no cover
            if hasattr(submodule, func_name):  # pragma: no cover
                return getattr(submodule, func_name)(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            f"{func_name} is not supported in the active backend."  # pragma: no cover
        )  # pragma: no cover
    raise NotImplementedError(f"{func_name} is not fully supported in tracing mode.")  # pragma: no cover
