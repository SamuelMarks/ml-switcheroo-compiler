"""Framework-agnostic object serialization and deserialization utilities."""

from __future__ import annotations

import typing
from collections.abc import Callable
from types import TracebackType

from ml_switcheroo_ir import AttributeValue

T = typing.TypeVar("T")


@typing.runtime_checkable
class ConfigurableProtocol(typing.Protocol):
    """Protocol for objects implementing get_config()."""

    def get_config(self) -> dict[str, AttributeValue]:
        """Return configuration dictionary.

        Returns:
            dict[str, AttributeValue]: Computed result.
        """
        ...


class SerializableCallableProtocol(typing.Protocol):
    """Protocol for serializable factory or function objects."""

    def __call__(
        self,
        *args: AttributeValue,
        **kwargs: AttributeValue,
    ) -> AttributeValue | ConfigurableProtocol:
        """Invoke the serializable callable.

        Args:
            *args (AttributeValue): Positional arguments.
            **kwargs (AttributeValue): Keyword arguments.

        Returns:
            AttributeValue | ConfigurableProtocol: Constructed or computed result.
        """
        ...


SerializableType = typing.Union[type, "SerializableCallableProtocol"]

_GLOBAL_CUSTOM_OBJECTS: dict[str, SerializableType] = {}
_REGISTERED_NAMES: dict[SerializableType, str] = {}


class CustomObjectScope:
    """Context manager for temporary scoped registration of custom serializable objects."""

    def __init__(
        self,
        *args: dict[str, SerializableType],
        **kwargs: SerializableType,
    ) -> None:
        """Initialize CustomObjectScope with dictionary or kwargs of custom objects.

        Args:
            *args (dict[str, SerializableType]): Mapping dictionary of custom objects.
            **kwargs (SerializableType): Key-value pairs of custom objects.
        """
        self.custom_objects: dict[str, SerializableType] = {}
        if args and isinstance(args[0], dict):
            self.custom_objects.update(args[0])
        self.custom_objects.update(kwargs)
        self.backup: dict[str, SerializableType] = {}

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
    *args: dict[str, SerializableType],
    **kwargs: SerializableType,
) -> CustomObjectScope:
    """Create a functional custom object scope context manager.

    Args:
        *args (dict[str, SerializableType]): Mapping dictionary of custom objects.
        **kwargs (SerializableType): Custom objects key-value pairs.

    Returns:
        CustomObjectScope: The context manager instance.
    """
    return CustomObjectScope(*args, **kwargs)


def get_custom_objects() -> dict[str, SerializableType]:
    """Retrieve the global registry of custom objects.

    Returns:
        dict[str, SerializableType]: Dictionary of registered custom objects.
    """
    return _GLOBAL_CUSTOM_OBJECTS


def get_registered_name(obj: SerializableType | None = None) -> str:
    """Get the registered serializable name of a class or function.

    Args:
        obj (SerializableType | None): Target class or function.

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
    custom_objects: dict[str, SerializableType] | None = None,
) -> SerializableType | None:
    """Retrieve registered object by name from custom objects or global registry.

    Args:
        name (str | None): Identifier name of the object.
        custom_objects (dict[str, SerializableType] | None): Optional custom objects to search first.

    Returns:
        SerializableType | None: Found class/function or None.
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
        registered_name = str(name or getattr(cls_or_fn, "__name__", str(cls_or_fn)))
        full_name = f"{package}>{registered_name}" if package else registered_name
        serializable_obj = typing.cast(SerializableType, cls_or_fn)
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = serializable_obj
        _GLOBAL_CUSTOM_OBJECTS[full_name] = serializable_obj
        _REGISTERED_NAMES[serializable_obj] = registered_name
        return cls_or_fn

    return decorator


def serialize_object(
    obj: ConfigurableProtocol | AttributeValue | None = None,
) -> dict[str, AttributeValue] | None:
    """Serialize a framework-agnostic object or layer into a configuration dictionary.

    Args:
        obj (ConfigurableProtocol | AttributeValue | None): Object to serialize.

    Returns:
        dict[str, AttributeValue] | None: Serialized dictionary or None.
    """
    if obj is None:
        return None
    cls_name = get_registered_name(obj.__class__)
    cfg: dict[str, AttributeValue] = obj.get_config() if hasattr(obj, "get_config") else {}
    return {
        "class_name": cls_name,
        "config": cfg,
        "module": getattr(obj.__class__, "__module__", ""),
        "registered_name": cls_name,
    }


def deserialize_object(
    identifier: dict[str, AttributeValue] | ConfigurableProtocol | AttributeValue | None = None,
    custom_objects: dict[str, SerializableType] | None = None,
) -> ConfigurableProtocol | AttributeValue | None:
    """Deserialize a configuration dictionary back into an object instance.

    Args:
        identifier (dict[str, AttributeValue] | ConfigurableProtocol | AttributeValue | None): Serialized config dict or object.
        custom_objects (dict[str, SerializableType] | None): Optional custom objects map.

    Returns:
        ConfigurableProtocol | AttributeValue | None: Deserialized object instance or original identifier.
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
            return typing.cast("ConfigurableProtocol | AttributeValue | None", cls_obj.from_config(config_dict))
        if isinstance(config_dict, dict):
            return typing.cast("ConfigurableProtocol | AttributeValue | None", cls_obj(**config_dict))
    return identifier
