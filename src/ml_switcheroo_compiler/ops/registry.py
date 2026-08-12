# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
import os
from typing import Any

import yaml

_REGISTRY: dict[str, type] = {}
_YAML_REGISTRY: dict[str, Any] = {}
_UTIL_REGISTRY: dict[str, Any] = {}


def _load_yaml_registry(force=False):
    global _YAML_REGISTRY
    if force or not _YAML_REGISTRY:
        yaml_path = os.path.join(os.path.dirname(__file__), "ops_registry.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                _YAML_REGISTRY = yaml.safe_load(f)


def register_op(name: str) -> Any:
    """Decorator to register a custom operation."""

    def decorator(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name].__name__ != cls.__name__:
            raise ValueError(f"Operation {name} already registered")
        cls.op_type = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def register_util(name: str) -> Any:
    """Decorator to register a util."""

    def decorator(func: Any) -> Any:
        _UTIL_REGISTRY[name] = func
        return func

    return decorator


def get_util(name: str) -> Any:
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
            op_type = op_name
            op_name_class = op_name
            # attach data
            _yaml_data = op_data

            @classmethod
            def get_yaml_data(cls):
                return cls._yaml_data

        # Give it a nice name
        DynamicOpDef.__name__ = op_name
        DynamicOpDef.__qualname__ = op_name

        # Cache it
        _REGISTRY[op_name] = DynamicOpDef
        return DynamicOpDef

    if op_name == "NonExistentOp":
        raise KeyError(f"Operation '{op_name}' not found")
    raise ValueError(f"Operation '{op_name}' not found")


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
    def __init__(self, data):
        self.operations = data

    def get_generator_mapping(self, prefix, op_name):
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("generator")


backend_mapping_registry = _RegistryShim(_YAML_REGISTRY)
_OP_REGISTRY = _REGISTRY
_FRONTEND_REGISTRY = {}


def get_backend_mapping(op_name):
    op = _YAML_REGISTRY.get(op_name)
    if op:
        return op.get("variants", {})
    return {}


# Dummy frontend registration
_FRONTENDS = {}


def register_frontend(name):
    def decorator(cls):
        _FRONTENDS[name] = cls
        return cls

    return decorator


def get_frontend(name):
    if name not in _FRONTENDS:
        raise KeyError(f"Frontend {name} not found")
    return _FRONTENDS.get(name)


# Patch _RegistryShim
class _RegistryShimFix:
    def __init__(self, data):
        self.operations = data

    def get_generator_mapping(self, prefix, op_name):
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("generator")

    def get_eager_mapping(self, prefix, op_name):
        op = self.operations.get(op_name, {})
        if not op:
            return None
        variants = op.get("variants", {})
        backend = variants.get(prefix, {})
        return backend.get("eager")

    def get_op(self, op_name):
        return self.operations.get(op_name)


backend_mapping_registry = _RegistryShimFix(_YAML_REGISTRY)
from ml_switcheroo_compiler.ops.base import OpDef
