"""WGSL template provider."""

from __future__ import annotations

import os
from typing import Any

import yaml

_WGSL_TEMPLATES: dict[str, dict[str, dict[str, str | list[int] | None]] | dict[str, str] | str | None] = {}
_WGSL_KERNELS: dict[str, object] = {}


def _load_kernels() -> dict[str, object]:
    """Load declarative WGSL kernels configuration from wgsl_kernels.yaml.

    Returns:
        dict[str, object]: Parsed and validated WGSL kernels configuration dictionary.
    """
    global _WGSL_KERNELS
    kernels_path = os.path.join(os.path.dirname(__file__), "wgsl_kernels.yaml")
    if not os.path.exists(kernels_path):
        return {"bindings": {}, "op_mappings": {}, "templates": {}}
    if not _WGSL_KERNELS:
        from ml_switcheroo_compiler.backends.edge.config_models import WgslKernelsConfig

        with open(kernels_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        if isinstance(raw_data, dict):
            _WGSL_KERNELS = WgslKernelsConfig(**raw_data).model_dump()
        if not _WGSL_KERNELS:
            _WGSL_KERNELS = {"bindings": {}, "op_mappings": {}, "templates": {}}
    return _WGSL_KERNELS


def get_wgsl_op_mapping(op_name: str) -> dict[str, str | int | float | None]:
    """Retrieve declarative WGSL mapping for an op.

    Args:
        op_name (str): Operation identifier.

    Returns:
        dict[str, Union[str, int, float, None]]: Mapping dictionary if defined, else empty dict.
    """
    kernels = _load_kernels()
    mappings = kernels.get("op_mappings", {})
    if isinstance(mappings, dict) and op_name in mappings:
        val = mappings[op_name]
        if isinstance(val, dict):
            return val
    return {}


def get_wgsl_kernels_config() -> dict[str, object]:
    """Retrieve the full WGSL kernels specification dictionary.

    Returns:
        dict[str, object]: Kernels config dictionary.
    """
    return _load_kernels()


def _merge_modular_templates(
    loaded_templates: dict[str, dict[str, dict[str, str | list[int] | None]] | dict[str, str] | str | None],
) -> None:
    """Merge modular WGSL templates from templates.yaml into loaded templates.

    Args:
        loaded_templates (dict): The target templates dictionary to update.
    """
    from ml_switcheroo_compiler.backends.edge.config_models import WgslTemplatesConfig

    modular_dir: str = os.path.join(os.path.dirname(__file__), "wgsl_templates")
    if os.path.exists(modular_dir):
        for fname in sorted(os.listdir(modular_dir)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(modular_dir, fname)
                try:
                    with open(fpath) as f:
                        mod_data = yaml.safe_load(f)
                    if isinstance(mod_data, dict):
                        mod_dump = WgslTemplatesConfig(**mod_data).model_dump()
                        loaded_templates.setdefault("templates", {}).update(mod_dump.get("templates") or {})
                except Exception:
                    pass

    modular_path: str = os.path.join(os.path.dirname(__file__), "templates.yaml")
    if not os.path.exists(modular_path):
        return
    with open(modular_path) as f:
        mod_data = yaml.safe_load(f)
    if not isinstance(mod_data, dict):
        return

    mod_dump = WgslTemplatesConfig(**mod_data).model_dump()
    loaded_templates.setdefault("templates", {}).update(mod_dump.get("templates") or {})
    loaded_templates.setdefault("js_orchestration", {}).update(mod_dump.get("js_orchestration") or {})
    if mod_dump.get("global_bindings"):
        loaded_templates["global_bindings"] = mod_dump["global_bindings"]


def _load_templates() -> None:
    """Load wgsl templates from YAML definitions."""
    global _WGSL_TEMPLATES
    if not _WGSL_TEMPLATES:
        from ml_switcheroo_compiler.backends.edge.config_models import WgslTemplatesConfig

        loaded: dict[str, dict[str, dict[str, str | list[int] | None]] | dict[str, str] | str | None] = {
            "templates": {},
            "js_orchestration": {},
            "global_bindings": None,
        }

        base_path: str = os.path.join(os.path.dirname(__file__), "wgsl_templates.yaml")
        if os.path.exists(base_path):
            with open(base_path) as f:
                base_data = yaml.safe_load(f)
                if isinstance(base_data, dict):
                    loaded = WgslTemplatesConfig(**base_data).model_dump()

        _merge_modular_templates(loaded)
        kernels = _load_kernels()
        if isinstance(kernels, dict):
            k_templates = kernels.get("templates")
            if isinstance(k_templates, dict):
                loaded.setdefault("templates", {}).update(k_templates)
            k_bindings = kernels.get("bindings")
            if isinstance(k_bindings, dict) and k_bindings.get("global_bindings") and not loaded.get("global_bindings"):
                loaded["global_bindings"] = k_bindings["global_bindings"]
        _WGSL_TEMPLATES = loaded


def get_wgsl_template(name: str) -> dict[str, Any]:
    """Retrieve WGSL template configuration by operation name.

    Args:
        name (str): Operation or template name.

    Returns:
        dict[str, Any]: Template configuration dict.
    """
    _load_templates()
    templates_dict = _WGSL_TEMPLATES.get("templates", {})
    if isinstance(templates_dict, dict):
        tpl = templates_dict.get(name, {})
        if isinstance(tpl, dict):
            return tpl
    return {}


def get_js_orchestration_template(name: str) -> str:
    """Retrieve JavaScript compute pass orchestration template by name.

    Args:
        name (str): Name of the orchestration snippet.

    Returns:
        str: JavaScript code snippet.
    """
    _load_templates()
    js_orch = _WGSL_TEMPLATES.get("js_orchestration", {})
    if isinstance(js_orch, dict):
        return str(js_orch.get(name, ""))
    return ""


def get_wgsl_global_bindings() -> str:
    """Retrieve WGSL global storage buffer bindings header string.

    Returns:
        str: WGSL declarations for storage buffer bind group 0.
    """
    _load_templates()
    val = _WGSL_TEMPLATES.get("global_bindings", "")
    return str(val) if val else ""


_WEBGPU_OPS: dict[str, str | dict[str, str] | dict[str, dict[str, str]]] = {}


def _load_webgpu_ops() -> None:
    """Load WebGPU operations dispatch and offset templates from YAML."""
    global _WEBGPU_OPS
    if not _WEBGPU_OPS:
        path: str = os.path.join(os.path.dirname(__file__), "webgpu_ops.yaml")
        if os.path.exists(path):
            with open(path) as f:
                raw_data = yaml.safe_load(f)
                _WEBGPU_OPS = raw_data if isinstance(raw_data, dict) else {}


def get_webgpu_ops() -> dict[str, str | dict[str, str] | dict[str, dict[str, str]]]:
    """Retrieve WebGPU operations dispatch rules and coordinate calculation templates.

    Returns:
        dict[str, Union[str, dict[str, str], dict[str, dict[str, str]]]]: Config map.
    """
    _load_webgpu_ops()
    return _WEBGPU_OPS


def get_bindgroup_schemas(path: str | None = None) -> dict[str, object]:
    """Retrieve WebGPU bind group configuration schemas dictionary.

    Args:
        path (str, optional): Custom path to bindgroup_schemas.yaml.

    Returns:
        dict[str, object]: Parsed bindgroup schemas dictionary.
    """
    from ml_switcheroo_compiler.backends.edge.config_models import load_bindgroup_schemas

    return load_bindgroup_schemas(path).model_dump()


def get_dtype_emulation_config(path: str | None = None) -> dict[str, object]:
    """Retrieve WGSL double-precision emulation configuration dictionary.

    Args:
        path (str, optional): Custom path to dtype_emulation.yaml.

    Returns:
        dict[str, object]: Parsed dtype emulation dictionary.
    """
    from ml_switcheroo_compiler.backends.edge.config_models import load_dtype_emulation_config

    return load_dtype_emulation_config(path).model_dump()


def generate_dynamic_bindgroup_declarations(
    num_inputs: int,
    num_outputs: int = 1,
    input_dtype: str = "f32",
    output_dtype: str = "f32",
) -> str:
    """Generate dynamic WGSL storage buffer bindings for arbitrary input and output counts.

    Args:
        num_inputs (int): Number of input storage buffers.
        num_outputs (int): Number of output storage buffers.
        input_dtype (str): Data type of input storage buffers.
        output_dtype (str): Data type of output storage buffers.

    Returns:
        str: WGSL declarations for bind group 0.
    """
    lines: list[str] = []
    # Bindings 0, 1, 2 for first 3 inputs
    for j in range(min(num_inputs, 3)):
        lines.append(f"@group(0) @binding({j}) var<storage, read> buf_in{j}_{input_dtype}: array<{input_dtype}>;")
    if num_inputs < 3:
        for j in range(num_inputs, 3):
            lines.append(f"@group(0) @binding({j}) var<storage, read> buf_in{j}_{input_dtype}: array<{input_dtype}>;")

    # Primary output at binding 3
    lines.append(f"@group(0) @binding(3) var<storage, read_write> buf_out_{output_dtype}: array<{output_dtype}>;")

    # Additional inputs (index 3 and higher) starting at binding 4
    for j in range(3, num_inputs):
        lines.append(f"@group(0) @binding({j + 1}) var<storage, read> buf_in{j}_{input_dtype}: array<{input_dtype}>;")

    # Additional outputs (index 1 and higher) starting at binding 6
    for k in range(1, num_outputs):
        lines.append(f"@group(0) @binding({5 + k}) var<storage, read_write> buf_out{k}_{output_dtype}: array<{output_dtype}>;")

    return "\n".join(lines)


_WGSL_GROUNDING_SCHEMA: dict[str, object] | None = None


def get_wgsl_grounding_schema() -> dict[str, object]:
    """Load and return the canonical WGSL grounding schema from ml_switcheroo_ir.schema.wgsl_ops.json.

    Returns:
        dict[str, object]: The parsed WGSL ops schema.
    """
    global _WGSL_GROUNDING_SCHEMA
    if _WGSL_GROUNDING_SCHEMA is not None:
        return _WGSL_GROUNDING_SCHEMA
    import json

    import ml_switcheroo_ir.schema

    schema_dir = os.path.dirname(ml_switcheroo_ir.schema.__file__)
    schema_path = os.path.join(schema_dir, "wgsl_ops.json")
    if os.path.exists(schema_path):
        with open(schema_path, encoding="utf-8") as f:
            _WGSL_GROUNDING_SCHEMA = json.load(f)
    else:
        _WGSL_GROUNDING_SCHEMA = {"ops": []}
    return _WGSL_GROUNDING_SCHEMA


def validate_wgsl_statement(op_name: str) -> bool:
    """Validate a generated WebGPU WGSL operation or statement against canonical wgsl_ops schema.

    Args:
        op_name (str): The WGSL op identifier to validate.

    Returns:
        bool: True if op is supported in the WGSL grounding schema or core templates.
    """
    schema = get_wgsl_grounding_schema()
    ops_list = schema.get("ops", [])
    if isinstance(ops_list, list):
        for op_entry in ops_list:
            if isinstance(op_entry, dict) and op_entry.get("name") == op_name:
                return True
    if bool(get_wgsl_op_mapping(op_name)):
        return True
    _load_templates()
    templates = _WGSL_TEMPLATES.get("templates", {})
    if isinstance(templates, dict):
        return op_name.lower() in templates or op_name in templates
    return False
