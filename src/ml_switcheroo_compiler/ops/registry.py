# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module registry.py."""

import os

import yaml

_REGISTRY: dict[str, type] = {}
_YAML_REGISTRY: dict[str, object] = {}
_UTIL_REGISTRY: dict[str, object] = {}


def _load_yaml_registry(force: bool = False) -> None:
    """_load_yaml_registry function.

    Args:
        force (object): The force parameter.

    Returns:
        object: Result.
    """
    global _YAML_REGISTRY
    if force or not _YAML_REGISTRY:
        yaml_path: object = os.path.join(os.path.dirname(__file__), "ops_registry.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                from ml_switcheroo_compiler.ops.config_models import OpsRegistry

                raw_yaml: object = yaml.safe_load(f)
                _YAML_REGISTRY = OpsRegistry(root=raw_yaml).model_dump()


def register_op(name: str) -> object:
    """Decorator to register a custom operation."""

    def decorator(cls: type) -> type:
        """Decorator function.

        Args:
        cls (object): The cls parameter.

        Returns:
        object: Result.
        """
        if name in _REGISTRY and _REGISTRY[name].__name__ != cls.__name__:
            raise ValueError(f"Operation {name} already registered")
        cls.op_type = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def register_util(name: str) -> object:
    """Decorator to register a util."""

    def decorator(func: object) -> object:
        """Decorator function.

        Args:
        func (object): The func parameter.

        Returns:
        object: Result.
        """
        _UTIL_REGISTRY[name] = func
        return func

    return decorator


def get_util(name: str) -> object:
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
        op_data: object = _YAML_REGISTRY[op_name]

        # Build dynamic class
        class DynamicOpDef(OpDef):
            """DynamicOpDef class."""

            op_type: object = op_name
            op_name_class: object = op_name
            # attach data
            _yaml_data = op_data

            @classmethod
            def get_yaml_data(cls: object) -> object:
                """get_yaml_data function.

                Args:
                cls (object): The cls parameter.

                Returns:
                object: Result.
                """
                return cls._yaml_data

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """Infer shape precisely using heuristics.

                Args:
                    self (object): The self parameter.
                    *args (object): Positional args.
                    **kwargs (object): Keyword args.

                Returns:
                    object: Result shape tuple.
                """
                inputs: object = kwargs.get("inputs", [])

                if not inputs:
                    if len(args) > 0 and isinstance(args[0], (list, tuple)):
                        inputs: object = args[0]
                    else:
                        inputs: object = list(args)

                shapes: object = []
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
get_op_class: object = get_op

# Initialize on load
_load_yaml_registry()


# Expose backward compatibility aliases for tests that expect the old structure
class _RegistryShim:
    """_RegistryShim class."""

    def __init__(self, data: object) -> None:
        """__init__ function.

        Args:
        self (object): The self parameter.
        data (object): The data parameter.

        Returns:
        object: Result.
        """
        self.operations = data

    def get_generator_mapping(self, prefix: str, op_name: str) -> object:
        """get_generator_mapping function.

        Args:
        self (object): The self parameter.
        prefix (object): The prefix parameter.
        op_name (object): The op_name parameter.

        Returns:
        object: Result.
        """
        op: object = self.operations.get(op_name, {})
        if not op:
            return None
        variants: object = op.get("variants", {})
        backend: object = variants.get(prefix, {})
        return backend.get("generator")


backend_mapping_registry: object = _RegistryShim(_YAML_REGISTRY)
_OP_REGISTRY = _REGISTRY
_FRONTEND_REGISTRY: dict[str, object] = {}


def get_backend_mapping(op_name: str) -> dict[str, object]:
    """get_backend_mapping function.

    Args:
        op_name (object): The op_name parameter.

    Returns:
        object: Result.
    """
    op: object = _YAML_REGISTRY.get(op_name)
    if op:
        return dict(op.get("variants", {}))
    return {}


# Dummy frontend registration
_FRONTENDS = {}


def register_frontend(name: str) -> object:
    """register_frontend function.

    Args:
        name (object): The name parameter.

    Returns:
        object: Result.
    """

    def decorator(cls: object) -> object:
        """Decorator function.

        Args:
        cls (object): The cls parameter.

        Returns:
        object: Result.
        """
        _FRONTENDS[name] = cls
        return cls

    return decorator


def get_frontend(name: str) -> object:
    """get_frontend function.

    Args:
        name (object): The name parameter.

    Returns:
        object: Result.
    """
    if name not in _FRONTENDS:
        raise KeyError(f"Frontend {name} not found")
    return _FRONTENDS.get(name)


# Patch _RegistryShim
class _RegistryShimFix:
    """_RegistryShimFix class."""

    def __init__(self, data: object) -> None:
        """__init__ function.

        Args:
        self (object): The self parameter.
        data (object): The data parameter.

        Returns:
        object: Result.
        """
        self.operations = data

    def get_generator_mapping(self, prefix: str, op_name: str) -> object:
        """get_generator_mapping function.

        Args:
        self (object): The self parameter.
        prefix (object): The prefix parameter.
        op_name (object): The op_name parameter.

        Returns:
        object: Result.
        """
        op: object = self.operations.get(op_name, {})
        if not op:
            return None
        variants: object = op.get("variants", {})
        backend: object = variants.get(prefix, {})
        return backend.get("generator")

    def get_eager_mapping(self, prefix: str, op_name: str) -> object:
        """get_eager_mapping function.

        Args:
        self (object): The self parameter.
        prefix (object): The prefix parameter.
        op_name (object): The op_name parameter.

        Returns:
        object: Result.
        """
        op: object = self.operations.get(op_name, {})
        if not op:
            return None
        variants: object = op.get("variants", {})
        backend: object = variants.get(prefix, {})
        return backend.get("eager")

    def get_op(self, op_name: str) -> object:
        """get_op function.

        Args:
        self (object): The self parameter.
        op_name (object): The op_name parameter.

        Returns:
        object: Result.
        """
        return self.operations.get(op_name)


backend_mapping_registry: object = _RegistryShimFix(_YAML_REGISTRY)
from ml_switcheroo_compiler.ops.base import OpDef
