"""Declarative and dynamic operation registry for the ML compiler."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.shape import CallableObjectProtocol, TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.ops.base import OpDef

_REGISTRY: dict[str, type[OpDef]] = {}
_YAML_REGISTRY: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
_UTIL_REGISTRY: dict[str, TypedVariadicCallable[TensorOrScalar]] = {}


def _load_yaml_registry(force: bool = False) -> None:
    """Load the declarative ops registry into memory without runtime parsing overhead.

    Args:
        force (bool): Whether to force reloading the registry.

    Returns:
        None: Populates _YAML_REGISTRY in-place.
    """
    global _YAML_REGISTRY
    if force or not _YAML_REGISTRY:
        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        _YAML_REGISTRY.clear()
        getattr(_YAML_REGISTRY, "update")(OPS_REGISTRY)


def register_op(name: str) -> Callable[[type], type]:
    """Decorator to register a custom operation.

    Args:
        name (str): Canonical operation name.

    Returns:
        Callable[[type], type]: Class decorator registering the operation.
    """

    def decorator(cls: type) -> type:
        """Register the operation class under the given name.

        Args:
            cls (type): The operation class to register.

        Returns:
            type: The registered operation class.
        """
        if name in _REGISTRY and _REGISTRY[name].__name__ != cls.__name__:
            raise ValueError(f"Operation {name} already registered")
        setattr(cls, "op_type", name)
        _REGISTRY[name] = cls
        return cls

    return decorator


F = typing.TypeVar("F", bound=CallableObjectProtocol)


def register_util(name: str) -> Callable[[F], F]:
    """Decorator to register a utility function.

    Args:
        name (str): Identifier name for the utility function.

    Returns:
        Callable[[F], F]: Decorator wrapping the target function.
    """

    def decorator(func: F) -> F:
        """Register the function in the utility registry.

        Args:
            func (F): The function to register.

        Returns:
            F: The registered function.
        """
        _UTIL_REGISTRY[name] = typing.cast(TypedVariadicCallable[TensorOrScalar], func)
        return func

    return decorator


def get_util(name: str) -> TypedVariadicCallable[TensorOrScalar]:
    """Retrieve a registered utility function by name.

    Args:
        name (str): Identifier of the utility function.

    Returns:
        TypedVariadicCallable[TensorOrScalar]: Registered utility callable.

    Raises:
        KeyError: If the utility function is not registered.
    """
    if name not in _UTIL_REGISTRY:
        raise KeyError(f"Util {name} not found")
    return _UTIL_REGISTRY[name]


def get_op(op_name: str) -> type[OpDef]:
    """Retrieve an operation class by name.

    Args:
        op_name (str): Canonical operation name.

    Returns:
        type[OpDef]: Operation definition class.

    Raises:
        KeyError: If the operation is a non-math metadata type or is not found.
    """
    from ml_switcheroo_compiler.diagnostics.types_registry import is_non_math_type

    if is_non_math_type(op_name):
        raise KeyError(f"'{op_name}' is a non-math metadata type/exception and not an executable operation.")

    if op_name in _REGISTRY:
        return _REGISTRY[op_name]

    _load_yaml_registry(force=False)

    if op_name in _YAML_REGISTRY:
        op_data = _YAML_REGISTRY[op_name]

        class DynamicOpDef(OpDef):
            """DynamicOpDef class synthesized declaratively."""

            op_type = op_name
            op_name_class = op_name
            signature = op_data.get("signature")
            dtype_rules = op_data.get("dtype_rules")
            shape_signature = op_data.get("shape_signature")
            invariants = op_data.get("invariants")
            autodiff = op_data.get("autodiff")
            variants = op_data.get("variants")
            _yaml_data = op_data

            @classmethod
            def get_yaml_data(cls) -> dict[str, dict[str, dict[str, str]]]:
                """Return raw declarative registry dictionary for this operation.

                Returns:
                    dict[str, dict[str, dict[str, str]]]: Raw YAML metadata dictionary.
                """
                return cls._yaml_data

            def infer_shape(self, *args: tuple[int, ...], **kwargs: int) -> tuple[int, ...]:
                """Infer shape declaratively using declarative shape signatures.

                Args:
                    *args (tuple[int, ...]): Input tensor shapes.
                    **kwargs (int): Shape inference keyword parameters.

                Returns:
                    tuple[int, ...]: Inferred output shape tuple.

                Raises:
                    ValueError: If input shapes violate the operation's declarative shape signature.
                """
                from ml_switcheroo_compiler.ops.shape_inference import infer_shape as _infer_shape

                return tuple(int(x) if isinstance(x, int) else 0 for x in _infer_shape(op_name, *args, **kwargs))

        DynamicOpDef.__name__ = op_name
        DynamicOpDef.__qualname__ = op_name

        _REGISTRY[op_name] = DynamicOpDef
        return DynamicOpDef

    raise KeyError(f"Operation '{op_name}' not found")


def get_all_ops() -> dict[str, type]:
    """Return all registered operations.

    Returns:
        dict[str, type]: Mapping from operation name to operation class.
    """
    _load_yaml_registry()
    for op_name in _YAML_REGISTRY:
        if op_name not in _REGISTRY:
            try:
                get_op(op_name)
            except KeyError:
                continue
    return _REGISTRY


get_op_class = get_op

_load_yaml_registry()


class _RegistryShim:
    """Backward-compatible registry shim exposing generator mappings."""

    def __init__(self, data: dict[str, dict[str, dict[str, dict[str, str]]]]) -> None:
        """Initialize the registry shim.

        Args:
            data (dict[str, dict[str, dict[str, dict[str, str]]]]): Operation registry dictionary.
        """
        self.operations = data

    def get_generator_mapping(self, prefix: str, op_name: str) -> str | None:
        """Retrieve generator mapping for a backend prefix and operation name.

        Args:
            prefix (str): Backend identifier prefix.
            op_name (str): Operation name.

        Returns:
            str | None: Generator mapping format string if present.
        """
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("generator")


_OP_REGISTRY = _REGISTRY
_FRONTEND_REGISTRY: dict[str, type] = {}


def get_backend_mapping(op_name: str) -> dict[str, dict[str, str]]:
    """Retrieve backend variants mapping for an operation.

    Args:
        op_name (str): Canonical operation name.

    Returns:
        dict[str, dict[str, str]]: Mapping of backend variants for the operation.
    """
    op = _YAML_REGISTRY.get(op_name)
    if op:
        return dict(op.get("variants", {}))
    return {}


_FRONTENDS: dict[str, type] = {}


def register_frontend(name: str) -> Callable[[type], type]:
    """Decorator to register a frontend class.

    Args:
        name (str): Frontend identifier name.

    Returns:
        Callable[[type], type]: Class decorator registering the frontend.
    """

    def decorator(cls: type) -> type:
        """Register the frontend class under the given name.

        Args:
            cls (type): Frontend class to register.

        Returns:
            type: The registered frontend class.
        """
        _FRONTENDS[name] = cls
        return cls

    return decorator


def get_frontend(name: str) -> type | None:
    """Retrieve a registered frontend class by name.

    Args:
        name (str): Frontend identifier name.

    Returns:
        type | None: Registered frontend class.

    Raises:
        KeyError: If the frontend is not registered.
    """
    if name not in _FRONTENDS:
        raise KeyError(f"Frontend {name} not found")
    return _FRONTENDS.get(name)


class _RegistryShimFix(_RegistryShim):
    """Extended registry shim supporting both generator and eager mappings."""

    def get_eager_mapping(self, prefix: str, op_name: str) -> str | None:
        """Retrieve eager mapping for a backend prefix and operation name.

        Args:
            prefix (str): Backend identifier prefix.
            op_name (str): Operation name.

        Returns:
            str | None: Eager mapping string if present.
        """
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("eager") or backend.get("generator") or backend.get("target_api")

    def get_op(self, op_name: str) -> dict[str, dict[str, dict[str, str]]] | None:
        """Retrieve raw operation entry from the registry.

        Args:
            op_name (str): Operation name.

        Returns:
            dict[str, dict[str, dict[str, str]]] | None: Operation dictionary if found.
        """
        return self.operations.get(op_name)


backend_mapping_registry = _RegistryShimFix(_YAML_REGISTRY)

__all__ = [
    "OpDef",
    "_FRONTEND_REGISTRY",
    "_FRONTENDS",
    "_OP_REGISTRY",
    "_REGISTRY",
    "_RegistryShim",
    "_RegistryShimFix",
    "_UTIL_REGISTRY",
    "_YAML_REGISTRY",
    "_load_yaml_registry",
    "backend_mapping_registry",
    "get_all_ops",
    "get_backend_mapping",
    "get_frontend",
    "get_op",
    "get_op_class",
    "get_util",
    "register_frontend",
    "register_op",
    "register_util",
]
