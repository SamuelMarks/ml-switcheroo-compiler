"""Custom serializable object registration and configuration (de)serialization."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue
from ml_switcheroo_compiler.serialization.protocols import _ConfigProviderProtocol, _FromConfigProtocol

T = typing.TypeVar("T", bound=OpValue)


@typing.runtime_checkable
class _NamedObjectProtocol(typing.Protocol):
    """Protocol for classes and functions exposing `__name__`."""

    @property
    def __name__(self) -> str:
        """Return object name.

        Returns:
            str: Object name.
        """
        ...


_CUSTOM_OBJECTS: dict[str, AttributeValue] = {}


def register_serializable(package: str = "Custom", name: str | None = None) -> Callable[[T], T]:
    """Register a custom object for serialization.

    Args:
        package (str): The package identifier under which to register.
        name (Optional[str]): Custom registered name.

    Returns:
        Callable[[T], T]: Decorator registering the target class or function.
    """

    def decorator(arg: T) -> T:
        """Register the given class or function.

        Args:
            arg (T): The class or function being decorated.

        Returns:
            T: The registered object unchanged.
        """
        raw_arg: OpValue = arg
        raw_name = str(raw_arg.__name__) if isinstance(raw_arg, _NamedObjectProtocol) else str(arg)
        registered_name = str(name or raw_name)
        _CUSTOM_OBJECTS[registered_name] = arg
        if package:
            _CUSTOM_OBJECTS[f"{package}>{registered_name}"] = arg
        return arg

    return decorator


class custom_object_scope:
    """Scope context manager for temporarily registering custom objects.

    Provides a scope in which custom objects are available for serialization and deserialization.
    """

    def __init__(self, *args: AttributeValue, **kwargs: AttributeValue) -> None:
        """Initialize custom_object_scope with custom object dictionary or kwargs.

        Args:
            *args (AttributeValue): Custom objects dictionary.
            **kwargs (AttributeValue): Custom objects as keyword arguments.
        """
        self.custom_objects = args[0] if args and isinstance(args[0], dict) else kwargs
        self._backup: dict[str, AttributeValue] = {}

    def __enter__(self) -> custom_object_scope:
        """Enter scope and register custom objects.

        Returns:
            custom_object_scope: The active context manager instance.
        """
        self._backup = dict(_CUSTOM_OBJECTS)
        _CUSTOM_OBJECTS.update(self.custom_objects)
        return self

    def __exit__(self, *args: AttributeValue, **kwargs: AttributeValue) -> None:
        """Exit scope and restore previous custom objects state.

        Args:
            *args (AttributeValue): Exception arguments if raised.
            **kwargs (AttributeValue): Keyword arguments.
        """
        _CUSTOM_OBJECTS.clear()
        _CUSTOM_OBJECTS.update(self._backup)


class CustomObjectScope(custom_object_scope):
    """Alias for custom_object_scope to maintain compatibility."""

    pass


def get_custom_objects(*args: AttributeValue, **kwargs: AttributeValue) -> dict[str, AttributeValue]:
    """Get the dictionary of currently registered custom objects.

    Args:
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        dict[str, AttributeValue]: Mapping of registered names to custom classes/functions.
    """
    return _CUSTOM_OBJECTS


def get_registered_name(obj: AttributeValue = None, *args: AttributeValue, **kwargs: AttributeValue) -> str:
    """Get the registered name for a given class or function.

    Args:
        obj (AttributeValue): Target class or function object.
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        str: The registered name of the object.
    """
    target: OpValue = obj if obj is not None else (args[0] if args else kwargs.get("obj"))
    inv_custom = {id(reg_obj): name for name, reg_obj in reversed(list(_CUSTOM_OBJECTS.items()))}
    if id(target) in inv_custom:
        return inv_custom[id(target)]
    if target is not None and isinstance(target, _NamedObjectProtocol):
        return str(target.__name__)
    return "CustomObject"


def get_registered_object(
    name: AttributeValue = None, *args: AttributeValue, **kwargs: AttributeValue
) -> AttributeValue:
    """Get the class or function registered under a specific name.

    Args:
        name (AttributeValue): Name identifier of the custom object.
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: The registered class or function object, or None if not found.
    """
    key = name if name is not None else (args[0] if args else kwargs.get("name"))
    return _CUSTOM_OBJECTS.get(str(key)) if key is not None else None


def _serialize_nested_config(val: AttributeValue) -> AttributeValue:
    """Recursively serialize nested values within configuration dictionaries.

    Args:
        val (AttributeValue): Config value to serialize.

    Returns:
        AttributeValue: Serialized primitive, dictionary, or list.
    """
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, _ConfigProviderProtocol):
        return serialize_object(val)
    if isinstance(val, dict):
        return {k: _serialize_nested_config(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_nested_config(item) for item in val]
    return val


def serialize_object(instance: AttributeValue = None, **kwargs: AttributeValue) -> AttributeValue:
    """Serialize an object or compiler config into its dictionary representation.

    Args:
        instance (AttributeValue): Object instance to serialize.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: Serialized dictionary structure.
    """
    obj = instance if instance is not None else (kwargs if kwargs else {})
    if not obj:
        return {}

    if isinstance(obj, _ConfigProviderProtocol):
        cfg = obj.get_config()
        obj_type = type(obj)
        inv_custom = {id(reg_obj): name for name, reg_obj in reversed(list(_CUSTOM_OBJECTS.items()))}
        reg_name: str | None = inv_custom.get(id(obj), inv_custom.get(id(obj_type)))

        if reg_name is not None:
            class_name = str(obj_type.__name__)
            mod_name: str | None = str(obj_type.__module__)
            return {
                "class_name": class_name,
                "config": _serialize_nested_config(cfg),
                "module": mod_name,
                "registered_name": reg_name,
            }
        return _serialize_nested_config(cfg)

    if isinstance(obj, dict):
        return {k: _serialize_nested_config(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_serialize_nested_config(item) for item in obj]

    return {}


def _try_instantiate_from_config(
    cls_type: AttributeValue, deserialized_sub: AttributeValue
) -> tuple[bool, AttributeValue]:
    """Attempt to instantiate `cls_type` via `.from_config(deserialized_sub)` if supported.

    Args:
        cls_type (AttributeValue): Registered class or callable.
        deserialized_sub (AttributeValue): Deserialized configuration payload.

    Returns:
        tuple[bool, AttributeValue]: `(True, instance)` if `.from_config` was used, else `(False, None)`.
    """
    cand_cls: AttributeValue | _FromConfigProtocol = cls_type
    if isinstance(cand_cls, _FromConfigProtocol):
        return True, cand_cls.from_config(deserialized_sub)
    return False, None


def _instantiate_registered_class(
    class_name: AttributeValue,
    sub_config: AttributeValue,
    custom_objects: dict[str, AttributeValue] | None,
) -> AttributeValue:
    """Instantiate a registered serializable class from its config payload.

    Args:
        class_name: Registered class identifier.
        sub_config: Nested configuration payload.
        custom_objects: Optional custom class mapping.

    Returns:
        AttributeValue: Instantiated object or class reference.

    Raises:
        ValueError: If class_name is not registered.
    """
    all_customs = dict(_CUSTOM_OBJECTS)
    if custom_objects:
        all_customs.update(custom_objects)

    if class_name not in all_customs:
        raise ValueError(
            f"Unknown class '{class_name}' encountered during deserialization. Register it via register_serializable."
        )

    cls_type = all_customs[str(class_name)]
    deserialized_sub = deserialize_object(sub_config, custom_objects=custom_objects)
    used_from_config, inst = _try_instantiate_from_config(cls_type, deserialized_sub)
    if used_from_config:
        return inst
    if callable(cls_type):
        return cls_type(**deserialized_sub) if isinstance(deserialized_sub, dict) else cls_type(deserialized_sub)
    return cls_type


def deserialize_object(
    config: AttributeValue = None,
    custom_objects: dict[str, AttributeValue] | None = None,
    **kwargs: AttributeValue,
) -> AttributeValue:
    """Deserialize an object or compiler config from its dictionary representation.

    Args:
        config (AttributeValue): Serialized configuration dictionary.
        custom_objects (Optional[dict[str, AttributeValue]]): Optional dictionary of registered custom classes.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: Deserialized object instance or reconstructed config dictionary.

    Raises:
        ValueError: If an unknown class identifier is encountered without registration.
    """
    cfg = config if config is not None else (kwargs if kwargs else {})
    if not cfg:
        return {}

    if isinstance(cfg, dict):
        if "class_name" in cfg and "config" in cfg:
            return _instantiate_registered_class(cfg["class_name"], cfg["config"], custom_objects)
        return {k: deserialize_object(v, custom_objects=custom_objects) for k, v in cfg.items()}

    if isinstance(cfg, (list, tuple)):
        return [deserialize_object(item, custom_objects=custom_objects) for item in cfg]

    return cfg
