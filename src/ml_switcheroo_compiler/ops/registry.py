"""Module registry.py."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import typing
from typing import Any, Callable

AnyType = typing.Union[int, float, str, bool, list, tuple, dict, None, type, typing.Callable]

import os

import yaml

_REGISTRY: dict[str, type] = {}
_YAML_REGISTRY = {}
_UTIL_REGISTRY = {}


def _load_yaml_registry(force: bool = False) -> None:
    """_load_yaml_registry function.

    Args:
        force (Any): The force parameter.

    Returns:
        Any: Result.
    """
    global _YAML_REGISTRY
    if force or not _YAML_REGISTRY:
        from ml_switcheroo_compiler.ops.config_models import OpsRegistry
        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        _YAML_REGISTRY.clear()
        _YAML_REGISTRY.update(OpsRegistry(root=OPS_REGISTRY).model_dump())


def register_op(name: str):
    """Decorator to register a custom operation."""

    def decorator(cls: type) -> type:
        """Decorator function.

        Args:
        cls (Any): The cls parameter.

        Returns:
        Any: Result.
        """
        if name in _REGISTRY and _REGISTRY[name].__name__ != cls.__name__:
            raise ValueError(f"Operation {name} already registered")
        cls.op_type = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def register_util(name: str):
    """Decorator to register a util."""

    def decorator(func):
        """Decorator function.

        Args:
        func (Any): The func parameter.

        Returns:
        Any: Result.
        """
        _UTIL_REGISTRY[name] = func
        return func

    return decorator


def get_util(name: str):
    """Get a util."""
    if name not in _UTIL_REGISTRY:
        raise KeyError(f"Util {name} not found")
    return _UTIL_REGISTRY[name]


def get_op(op_name: str) -> type:
    """Retrieve an operation class by name."""
    if op_name in _REGISTRY:
        return _REGISTRY[op_name]

    _load_yaml_registry(force=False)

    if op_name in _YAML_REGISTRY:
        from ml_switcheroo_compiler.ops.base import OpDef

        # Create dynamic OpDef class from yaml
        op_data = _YAML_REGISTRY[op_name]

        # Build dynamic class
        class DynamicOpDef(OpDef):
            """DynamicOpDef class."""

            op_type = op_name
            op_name_class = op_name
            # attach data
            _yaml_data = op_data

            @classmethod
            def get_yaml_data(cls):
                """get_yaml_data function.

                Args:
                cls (Any): The cls parameter.

                Returns:
                Any: Result.
                """
                return cls._yaml_data

            def infer_shape(self, *args, **kwargs):
                """Infer shape precisely using heuristics.

                Args:
                    self (Any): The self parameter.
                    *args (Any): Positional args.
                    **kwargs (Any): Keyword args.

                Returns:
                    Any: Result shape tuple.
                """
                inputs = kwargs.get("inputs", [])

                if not inputs:
                    if len(args) > 0 and isinstance(args[0], (list, tuple)):
                        inputs = args[0]
                    else:
                        inputs = list(args)

                shapes = []
                for inp in inputs:
                    if hasattr(inp, "shape_metadata") and inp.shape_metadata is not None:
                        shapes.append(tuple(inp.shape_metadata))
                    elif hasattr(inp, "shape") and inp.shape is not None:
                        shapes.append(tuple(inp.shape))
                    elif isinstance(inp, (list, tuple)) and all(isinstance(x, int) for x in inp):
                        shapes.append(tuple(inp))

                if not shapes:
                    return ()

                if len(shapes) == 1:
                    return shapes[0]

                try:
                    import numpy as np

                    return np.broadcast_shapes(*shapes)
                except Exception:
                    return max(shapes, key=len)

        # Give it a nice name
        DynamicOpDef.__name__ = op_name
        DynamicOpDef.__qualname__ = op_name

        # Cache it
        _REGISTRY[op_name] = DynamicOpDef
        return DynamicOpDef

    raise KeyError(f"Operation '{op_name}' not found")


def get_all_ops() -> dict[str, type]:
    """Return all registered operations."""
    _load_yaml_registry()
    for op_name in _YAML_REGISTRY:
        if op_name not in _REGISTRY:
            get_op(op_name)
    return _REGISTRY


# Alias for backwards compatibility
get_op_class = get_op

# Initialize on load
_load_yaml_registry()


# Expose backward compatibility aliases for tests that expect the old structure
class _RegistryShim:
    """_RegistryShim class."""

    def __init__(self, data) -> None:
        """__init__ function.

        Args:
        self (Any): The self parameter.
        data (Any): The data parameter.

        Returns:
        Any: Result.
        """
        self.operations = data

    def get_generator_mapping(self, prefix: str, op_name: str):
        """get_generator_mapping function.

        Args:
        self (Any): The self parameter.
        prefix (Any): The prefix parameter.
        op_name (Any): The op_name parameter.

        Returns:
        Any: Result.
        """
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("generator")


backend_mapping_registry = _RegistryShim(_YAML_REGISTRY)
_OP_REGISTRY = _REGISTRY
_FRONTEND_REGISTRY = {}


def get_backend_mapping(op_name: str):
    """get_backend_mapping function.

    Args:
        op_name (Any): The op_name parameter.

    Returns:
        Any: Result.
    """
    op = _YAML_REGISTRY.get(op_name)
    if op:
        return dict(op.get("variants", {}))
    return {}


# Frontend registration
_FRONTENDS = {}


def register_frontend(name: str):
    """register_frontend function.

    Args:
        name (Any): The name parameter.

    Returns:
        Any: Result.
    """

    def decorator(cls):
        """Decorator function.

        Args:
        cls (Any): The cls parameter.

        Returns:
        Any: Result.
        """
        _FRONTENDS[name] = cls
        return cls

    return decorator


def get_frontend(name: str):
    """get_frontend function.

    Args:
        name (Any): The name parameter.

    Returns:
        Any: Result.
    """
    if name not in _FRONTENDS:
        raise KeyError(f"Frontend {name} not found")
    return _FRONTENDS.get(name)


# Patch _RegistryShim
class _RegistryShimFix:
    """_RegistryShimFix class."""

    def __init__(self, data) -> None:
        """__init__ function.

        Args:
        self (Any): The self parameter.
        data (Any): The data parameter.

        Returns:
        Any: Result.
        """
        self.operations = data

    def get_generator_mapping(self, prefix: str, op_name: str):
        """get_generator_mapping function.

        Args:
        self (Any): The self parameter.
        prefix (Any): The prefix parameter.
        op_name (Any): The op_name parameter.

        Returns:
        Any: Result.
        """
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("generator")

    def get_eager_mapping(self, prefix: str, op_name: str):
        """get_eager_mapping function.

        Args:
        self (Any): The self parameter.
        prefix (Any): The prefix parameter.
        op_name (Any): The op_name parameter.

        Returns:
        Any: Result.
        """
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("eager")

    def get_op(self, op_name: str):
        """get_op function.

        Args:
        self (Any): The self parameter.
        op_name (Any): The op_name parameter.

        Returns:
        Any: Result.
        """
        return self.operations.get(op_name)


backend_mapping_registry = _RegistryShimFix(_YAML_REGISTRY)
from ml_switcheroo_compiler.ops.base import OpDef
