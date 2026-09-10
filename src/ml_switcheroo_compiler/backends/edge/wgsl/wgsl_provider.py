"""WGSL template provider."""

import os
from typing import Any, Union

import yaml

_WGSL_TEMPLATES: dict[str, Union[dict[str, dict[str, Union[str, list[int], None]]], dict[str, str], str, None]] = {}
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


def get_wgsl_op_mapping(op_name: str) -> dict[str, Union[str, int, float, None]]:
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
    loaded_templates: dict[str, Union[dict[str, dict[str, Union[str, list[int], None]]], dict[str, str], str, None]],
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

        loaded: dict[str, Union[dict[str, dict[str, Union[str, list[int], None]]], dict[str, str], str, None]] = {
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


_WEBGPU_OPS: dict[str, Union[str, dict[str, str], dict[str, dict[str, str]]]] = {}


def _load_webgpu_ops() -> None:
    """Load WebGPU operations dispatch and offset templates from YAML."""
    global _WEBGPU_OPS
    if not _WEBGPU_OPS:
        path: str = os.path.join(os.path.dirname(__file__), "webgpu_ops.yaml")
        if os.path.exists(path):
            with open(path) as f:
                raw_data = yaml.safe_load(f)
                _WEBGPU_OPS = raw_data if isinstance(raw_data, dict) else {}


def get_webgpu_ops() -> dict[str, Union[str, dict[str, str], dict[str, dict[str, str]]]]:
    """Retrieve WebGPU operations dispatch rules and coordinate calculation templates.

    Returns:
        dict[str, Union[str, dict[str, str], dict[str, dict[str, str]]]]: Config map.
    """
    _load_webgpu_ops()
    return _WEBGPU_OPS
