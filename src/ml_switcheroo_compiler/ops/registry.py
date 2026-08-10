# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
import typing
from typing import Any

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
        _OP_REGISTRY[name] = cls  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        cls.op_type = name  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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


# --- NEW BACKEND MAPPING REGISTRY ---
from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY


class BackendRegistry:
    """Central registry for all backend generator mappings."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.operations: dict[str, dict[str, Any]] = OPS_REGISTRY

    def get_op(self, op_name: str) -> typing.Optional[dict[str, Any]]:
        """Retrieve operation definition by name.

        Args:
            op_name (str): The name of the operation.

        Returns:
            Optional[dict[str, Any]]: The operation dict or None.
        """
        return self.operations.get(op_name)

    def get_eager_mapping(self, backend: str, op_name: str) -> typing.Optional[str]:
        """Retrieve eager execution string for a specific backend and operation.

        Args:
            backend (str): The name of the backend.
            op_name (str): The name of the operation.

        Returns:
            Optional[str]: The eager execution string or None.
        """
        op = self.get_op(op_name)
        if op and "variants" in op and backend in op["variants"]:
            return op["variants"][backend].get("eager")
        return None

    def get_generator_mapping(self, backend: str, op_name: str) -> typing.Optional[str]:
        """Retrieve generator format string for a specific backend and operation.

        Args:
            backend (str): The name of the backend.
            op_name (str): The name of the operation.

        Returns:
            Optional[str]: The generator format string or None.
        """
        op = self.get_op(op_name)
        if op and "variants" in op and backend in op["variants"]:
            return op["variants"][backend].get("generator")
        return None


backend_mapping_registry = BackendRegistry()
