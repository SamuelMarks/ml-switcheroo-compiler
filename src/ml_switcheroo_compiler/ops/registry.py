"""Central registry for all operations to break dependency cycles."""

from typing import Callable, TypeVar

from ml_switcheroo_compiler.ops.base import OpDef

T = TypeVar("T")
_OP_REGISTRY: dict[str, type["OpDef"]] = {}
_UTIL_REGISTRY: dict[str, Callable] = {}
_FRONTEND_REGISTRY: dict[str, Callable] = {}


def register_op(name: str) -> Callable[[type[T]], type[T]]:
    """Register op.

    Args:
        name (str): The name parameter.

    Returns:
        Callable: Result.
    """

    def decorator(cls: type[T]) -> type[T]:
        """Evaluate decorator operation.

        Args:
            cls (object): The cls parameter.

        Returns:
            type: Result.
        """
        if name in _OP_REGISTRY and _OP_REGISTRY[name].__name__ != cls.__name__:
            msg = f"Operation '{name}' is already registered."
            raise ValueError(msg)
        _OP_REGISTRY[name] = cls
        cls.op_type = name
        return cls

    return decorator


def register_util(name: str) -> Callable:
    """Register util.

    Args:
        name (str): name
    Returns:
        Callable: decorator
    """

    def decorator(func: Callable) -> Callable:
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        _UTIL_REGISTRY[name] = func
        return func

    return decorator


def get_util(name: str) -> Callable:
    """Retrieve a util function by name.

    Args:
        name (str): The name parameter.

    Returns:
        Callable: Result.

    Raises:
        KeyError: An exception.
    """
    if name not in _UTIL_REGISTRY:
        msg = f"Util '{name}' not found in registry."
        raise KeyError(msg)
    return _UTIL_REGISTRY[name]


def register_frontend(name: str) -> Callable:
    """Register frontend function.

    Args:
        name (str): name
    Returns:
        Callable: decorator
    """

    def decorator(func: Callable) -> Callable:
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        _FRONTEND_REGISTRY[name] = func
        return func

    return decorator


def get_frontend(name: str) -> Callable:
    """Retrieve a frontend function by name.

    Args:
        name (str): The name parameter.

    Returns:
        Callable: Result.

    Raises:
        KeyError: An exception.
    """
    if name not in _FRONTEND_REGISTRY:
        msg = f"Frontend '{name}' not found in registry."
        raise KeyError(msg)
    return _FRONTEND_REGISTRY[name]


def get_op(name: str) -> type["OpDef"]:
    """Retrieve an operation class by name.

    Args:
        name (str): The name parameter.

    Returns:
        type: Result.

    Raises:
        KeyError: An exception.
    """
    if name not in _OP_REGISTRY:
        msg = f"Operation '{name}' not found in registry."
        raise KeyError(msg)
    return _OP_REGISTRY[name]
