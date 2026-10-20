"""Backend YAML Mapping Loader and Dynamic Eager Dispatch."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Sequence

import yaml
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import AttrValue as AttributeValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


YamlDefaultValue = typing.Union[int, float, str, bool, list[typing.Union[int, float, str, bool, "None"]], dict[str, typing.Union[int, float, str, bool, "None"]], "None"]


class KwargTranslation(BaseModel):
    """Configuration for keyword argument translation between frameworks."""

    target_name: str
    default_value: YamlDefaultValue | None = None


class OpMappingSchema(BaseModel):
    """Schema defining mapping rules for a single operation."""

    operation: str | None = None
    backend: str | None = None
    target_api: str = ""
    is_method: bool = False
    kwarg_map: dict[str, str | None] = Field(default_factory=dict)
    kwarg_translations: dict[str, str] = Field(default_factory=dict)
    dtype_overrides: dict[str, str] = Field(default_factory=dict)
    default_kwargs: dict[str, YamlDefaultValue] = Field(default_factory=dict)
    supported_dtypes: list[str] | None = None
    ast_template: str | None = None
    custom_code: str | None = None


class BackendMappingSchema(BaseModel):
    """Schema representing complete mapping specifications for a backend."""

    backend_name: str
    operations: dict[str, OpMappingSchema]
    helpers: list[str] | None = None


_MAPPING_CACHE: dict[str, BackendMappingSchema] = {}


def _read_and_merge(path: str, target_dict: dict[str, AttributeValue]) -> None:
    """Read a YAML file and merge its operation specifications into target_dict.

    Args:
        path (str): The file path to load.
        target_dict (dict[str, object]): Target dictionary to update.
    """
    with open(path, encoding="utf-8") as f:
        data: dict[str, AttributeValue] = yaml.safe_load(f) or {}
    if isinstance(data, dict):
        if "operation" in data and isinstance(data["operation"], str):
            target_dict[data["operation"]] = data
        else:
            ops = data.get("operations")
            if isinstance(ops, dict):
                target_dict.update(ops)
            else:
                target_dict.update(data)


def _load_yaml_dir(yaml_dir: str, target_dict: dict[str, AttributeValue]) -> None:
    """Load all yaml files in directory into target_dict.

    Args:
        yaml_dir (str): Directory containing YAML files.
        target_dict (dict[str, object]): Target dictionary to update.
    """
    if not os.path.isdir(yaml_dir):
        return
    for filename in os.listdir(yaml_dir):
        if filename.endswith(".yaml"):
            file_path = os.path.join(yaml_dir, filename)
            _read_and_merge(file_path, target_dict)


def load_backend_mappings(backend_name: str) -> BackendMappingSchema:
    """Load backend operation mappings from declarative YAML files.

    Args:
        backend_name (str): The name of the target backend.

    Returns:
        BackendMappingSchema: Parsed backend mapping schema.
    """
    global _MAPPING_CACHE
    if backend_name in _MAPPING_CACHE:
        return _MAPPING_CACHE[backend_name]

    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    operations_dict: dict[str, AttributeValue] = {}

    _load_yaml_dir(os.path.join(base_dir, backend_name, "mappings"), operations_dict)

    yaml_path = os.path.join(base_dir, backend_name, "mappings.yaml")
    if os.path.exists(yaml_path):
        _read_and_merge(yaml_path, operations_dict)

    eager_yaml_path = os.path.join(base_dir, backend_name, "eager_mappings.yaml")
    if os.path.exists(eager_yaml_path):
        _read_and_merge(eager_yaml_path, operations_dict)

    schema = BackendMappingSchema.model_validate({"backend_name": backend_name, "operations": operations_dict})
    _MAPPING_CACHE[backend_name] = schema
    return schema


def _resolve_custom_code(custom_code: str, backend_module: AttributeValue | None) -> AttributeValue | None:
    """Evaluate custom code expression in a restricted namespace.

    Args:
        custom_code (str): Code string to evaluate.
        backend_module (Optional[object]): Module container for context.

    Returns:
        Optional[object]: Evaluated callable or None.
    """
    local_env: dict[str, AttributeValue] = {"backend_module": backend_module}
    if backend_module:
        for k in dir(backend_module):
            if not k.startswith("__"):
                try:
                    local_env[k] = getattr(backend_module, k)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    try:
        return typing.cast("AttributeValue | None", eval(custom_code, local_env))
    except Exception:
        return None


def _resolve_by_import(api_str: str) -> AttributeValue | None:
    """Import and traverse attribute chain.

    Args:
        api_str (str): Dot-separated module and attribute path.

    Returns:
        Optional[object]: Resolved object or None.
    """
    parts: list[str] = api_str.split(".")
    try:
        import importlib

        mod: AttributeValue = importlib.import_module(parts[0])
        for p in parts[1:]:
            mod = getattr(mod, p)
        return mod
    except Exception:
        return None


def _resolve_from_backend_module(api_str: str, backend_module: AttributeValue) -> AttributeValue | None:
    """Traverse attribute chain on backend module.

    Args:
        api_str (str): Target API string.
        backend_module (AttributeValue): Backend module.

    Returns:
        Optional[object]: Resolved function or None.
    """
    parts: list[str] = api_str.split(".")
    if len(parts) > 2 and parts[0] == "dask" and parts[1] == "array":
        sub_parts: list[str] = parts[2:]
    elif len(parts) > 1 and parts[0] in ("cupy", "dask", "tensorflow", "keras", "torch", "jax", "numpy", "np", "cp", "tf", "da", "numba", "nb", "sparse", "sp"):
        sub_parts = parts[1:]
    else:
        sub_parts = parts
    curr: AttributeValue = backend_module
    for p in sub_parts:
        if hasattr(curr, p):
            curr = getattr(curr, p)
        else:
            return getattr(backend_module, parts[-1], None) if hasattr(backend_module, parts[-1]) else None
    return curr


def resolve_target_api(
    api_str: str,
    custom_code: str | None = None,
    backend_module: AttributeValue | None = None,
) -> AttributeValue | None:
    """Resolve a target function reference from module path or custom code snippet.

    Args:
        api_str (str): Qualified import string or 'custom_op'.
        custom_code (Optional[str]): Lambda expression or snippet to evaluate.
        backend_module (Optional[object]): Module or mock container to bind globals.

    Returns:
        Optional[object]: Resolved callable function, or None if unresolvable.
    """
    if api_str == "custom_op" and custom_code:
        return _resolve_custom_code(custom_code, backend_module)

    if not api_str:
        return None

    if backend_module is not None:
        mod_res = _resolve_from_backend_module(api_str, backend_module)
        if mod_res is not None:
            return mod_res

    res = _resolve_by_import(api_str)
    if res is not None:
        return res

    return getattr(backend_module, api_str, None) if backend_module else None


def translate_kwargs(kwarg_translations: dict[str, str], kwargs: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
    """Translate caller keyword arguments according to backend schema mapping.

    Args:
        kwarg_translations (dict[str, str]): Mapping from source kwarg name to target name.
        kwargs (dict[str, object]): Keyword arguments provided by caller.

    Returns:
        dict[str, object]: Keyword arguments mapped to target framework parameter names.
    """
    translated: dict[str, AttributeValue] = {}
    for k, v in kwargs.items():
        target_k = kwarg_translations.get(k, k)
        translated[target_k] = v
    return translated


def dispatch_eager_op(
    backend_name: str,
    op_type: str,
    args: Sequence[AttributeValue | EagerValue],
    kwargs: dict[str, AttributeValue | EagerValue],
    backend_module: AttributeValue | EagerValue | None = None,
) -> EagerValue:
    """Dispatch an eager operation dynamically via declarative YAML mapping.

    Args:
        backend_name (str): The backend identifier (e.g., 'cupy', 'dask', 'tensorflow', 'keras').
        op_type (str): The logical operation name.
        args (Sequence[object]): Positional arguments for the op.
        kwargs (dict[str, object]): Keyword arguments for the op.
        backend_module (Optional[object]): Module or mock to bind custom expressions.

    Returns:
        EagerValue: Result of evaluating the mapped operation.

    Raises:
        BackendNotSupportedError: If op_type is unmapped or cannot be resolved.
    """
    schema = load_backend_mappings(backend_name)
    if op_type not in schema.operations:
        raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by {backend_name} eager backend.")

    op_spec = schema.operations[op_type]
    func = resolve_target_api(op_spec.target_api, op_spec.custom_code, backend_module)
    if func is None or not callable(func):
        raise BackendNotSupportedError(f"Operation '{op_type}' target API '{op_spec.target_api}' could not be resolved for {backend_name}.")

    kwarg_trans: dict[str, str] = dict(getattr(op_spec, "kwarg_translations", {}) or {})
    kwarg_map: dict[str, str | None] = getattr(op_spec, "kwarg_map", {}) or {}
    for src_k, dst_k in kwarg_map.items():
        if dst_k and src_k not in kwarg_trans:
            kwarg_trans[src_k] = dst_k
    translated_kwargs = translate_kwargs(kwarg_trans, kwargs)
    default_kwargs = getattr(op_spec, "default_kwargs", {}) or {}
    for def_k, def_v in default_kwargs.items():
        if def_k not in translated_kwargs:
            translated_kwargs[def_k] = def_v

    if getattr(op_spec, "is_method", False) and len(args) > 0:
        method = getattr(args[0], op_spec.target_api, None)
        if method is not None and callable(method):
            return typing.cast("EagerValue", method(*args[1:], **translated_kwargs))

    return func(*args, **translated_kwargs)
