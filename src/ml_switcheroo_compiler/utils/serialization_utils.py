"""Framework-agnostic object serialization and deserialization utilities."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

T = TypeVar("T")

_GLOBAL_CUSTOM_OBJECTS: dict[str, type | Callable[..., object]] = {}
_REGISTERED_NAMES: dict[type | Callable[..., object], str] = {}


class CustomObjectScope:
    """Context manager for temporary scoped registration of custom serializable objects."""

    def __init__(
        self,
        *args: dict[str, type | Callable[..., object]],
        **kwargs: type | Callable[..., object],
    ) -> None:
        """Initialize CustomObjectScope with dictionary or kwargs of custom objects.

        Args:
            *args (dict[str, type | Callable[..., object]]): Mapping dictionary of custom objects.
            **kwargs (type | Callable[..., object]): Key-value pairs of custom objects.
        """
        self.custom_objects: dict[str, type | Callable[..., object]] = {}
        if args and isinstance(args[0], dict):
            self.custom_objects.update(args[0])
        self.custom_objects.update(kwargs)
        self.backup: dict[str, type | Callable[..., object]] = {}

    def __enter__(self) -> CustomObjectScope:
        """Enter scope and merge custom objects into global registry.

        Returns:
            CustomObjectScope: The active scope context manager.
        """
        self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
        _GLOBAL_CUSTOM_OBJECTS.update(self.custom_objects)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit scope and restore previous global custom object state.

        Args:
            exc_type (type[BaseException] | None): Exception type if raised.
            exc_val (BaseException | None): Exception instance if raised.
            exc_tb (TracebackType | None): Traceback if raised.
        """
        _GLOBAL_CUSTOM_OBJECTS.clear()
        _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


def custom_object_scope(
    *args: dict[str, type | Callable[..., object]],
    **kwargs: type | Callable[..., object],
) -> CustomObjectScope:
    """Create a functional custom object scope context manager.

    Args:
        *args (dict[str, type | Callable[..., object]]): Mapping dictionary of custom objects.
        **kwargs (type | Callable[..., object]): Custom objects key-value pairs.

    Returns:
        CustomObjectScope: The context manager instance.
    """
    return CustomObjectScope(*args, **kwargs)


def get_custom_objects() -> dict[str, type | Callable[..., object]]:
    """Retrieve the global registry of custom objects.

    Returns:
        dict[str, type | Callable[..., object]]: Dictionary of registered custom objects.
    """
    return _GLOBAL_CUSTOM_OBJECTS


def get_registered_name(obj: type | Callable[..., object] | None = None) -> str:
    """Get the registered serializable name of a class or function.

    Args:
        obj (type | Callable[..., object] | None): Target class or function.

    Returns:
        str: Registered name string or empty string.
    """
    if obj is None:
        return ""
    if obj in _REGISTERED_NAMES:
        return _REGISTERED_NAMES[obj]
    if hasattr(obj, "__name__"):
        return str(obj.__name__)
    return str(type(obj).__name__)


def get_registered_object(
    name: str | None = None,
    custom_objects: dict[str, type | Callable[..., object]] | None = None,
) -> type | Callable[..., object] | None:
    """Retrieve registered object by name from custom objects or global registry.

    Args:
        name (str | None): Identifier name of the object.
        custom_objects (dict[str, type | Callable[..., object]] | None): Optional custom objects to search first.

    Returns:
        type | Callable[..., object] | None: Found class/function or None.
    """
    if not name:
        return None
    if custom_objects and name in custom_objects:
        return custom_objects[name]
    if name in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[name]
    return None


def register_serializable(
    package: str = "Custom",
    name: str | None = None,
) -> Callable[[T], T]:
    """Register a class or function as framework-agnostic serializable.

    Args:
        package (str): Package namespace name. Defaults to "Custom".
        name (str | None): Optional explicit registration name.

    Returns:
        Callable[[T], T]: Class/function decorator.
    """

    def decorator(cls_or_fn: T) -> T:
        """Register the annotated class in the global registry.

        Args:
            cls_or_fn (T): The class or function to register.

        Returns:
            T: The original class or function.
        """
        registered_name = name or getattr(cls_or_fn, "__name__", str(cls_or_fn))
        full_name = f"{package}>{registered_name}" if package else registered_name
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = cls_or_fn
        _GLOBAL_CUSTOM_OBJECTS[full_name] = cls_or_fn
        _REGISTERED_NAMES[cls_or_fn] = registered_name
        return cls_or_fn

    return decorator


def serialize_object(
    obj: object | None = None,
) -> dict[str, object] | None:
    """Serialize a framework-agnostic object or layer into a configuration dictionary.

    Args:
        obj (object | None): Object to serialize.

    Returns:
        dict[str, object] | None: Serialized dictionary or None.
    """
    if obj is None:
        return None
    cls_name = get_registered_name(obj.__class__)
    cfg: dict[str, object] = obj.get_config() if hasattr(obj, "get_config") else {}
    return {
        "class_name": cls_name,
        "config": cfg,
        "module": getattr(obj.__class__, "__module__", ""),
        "registered_name": cls_name,
    }


def deserialize_object(
    identifier: object | None = None,
    custom_objects: dict[str, type | Callable[..., object]] | None = None,
) -> object | None:
    """Deserialize a configuration dictionary back into an object instance.

    Args:
        identifier (object | None): Serialized config dict or object.
        custom_objects (dict[str, type | Callable[..., object]] | None): Optional custom objects map.

    Returns:
        object | None: Deserialized object instance or original identifier.
    """
    if identifier is None:
        return None
    if not isinstance(identifier, dict):
        return identifier
    class_name = identifier.get("class_name")
    if not isinstance(class_name, str):
        return identifier
    config_dict = identifier.get("config", {})
    cls_obj = get_registered_object(class_name, custom_objects)
    if cls_obj is not None and isinstance(cls_obj, type):
        if hasattr(cls_obj, "from_config") and isinstance(config_dict, dict):
            return cls_obj.from_config(config_dict)
        if isinstance(config_dict, dict):
            return cls_obj(**config_dict)
    return identifier


# Backward compatibility aliases
register_keras_serializable = register_serializable
serialize_keras_object = serialize_object
deserialize_keras_object = deserialize_object
serialize_keras_Any = serialize_object
deserialize_keras_Any = deserialize_object
custom_Any_scope = custom_object_scope
get_custom_Anys = get_custom_objects
get_registered_Any = get_registered_object
