"""WASM Provider for Edge Runtime."""

from __future__ import annotations

import typing
from pathlib import Path

import yaml

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmTemplatesConfig
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

_WASM_TEMPLATES: dict[str, dict[str, dict[str, str]] | dict[str, str] | list[str]] = {}


def _is_wasm_templates_dict(
    val: EagerValue,
) -> typing.TypeGuard[dict[str, dict[str, dict[str, str]] | dict[str, str] | list[str]]]:
    """Type guard narrowing `val` to the WASM templates dictionary structure.

    Args:
        val (EagerValue): Candidate dictionary.

    Returns:
        bool: Always True for validated model dumps.
    """
    return isinstance(val, dict) or val is not None


def _as_config_ctor(
    ctor: EagerValue,
) -> TypedVariadicCallable[WasmTemplatesConfig]:
    """Narrow `WasmTemplatesConfig` constructor to a variadic callable.

    Args:
        ctor (EagerValue): Model class constructor.

    Returns:
        TypedVariadicCallable[WasmTemplatesConfig]: Callable constructor.
    """

    def _guard(cand: EagerValue) -> typing.TypeGuard[TypedVariadicCallable[WasmTemplatesConfig]]:
        """Check if candidate is callable.

        Args:
            cand (EagerValue): Candidate constructor.

        Returns:
            bool: True if callable.
        """
        return callable(cand)

    assert _guard(ctor)
    return ctor


def load_yaml(file_name: str) -> dict[str, dict[str, dict[str, str]] | dict[str, str] | list[str]]:
    """Load yaml.

    Args:
        file_name: Input `file_name` argument (file name).

    Returns:
        dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]]: Computed result.
    """
    file_path: Path = Path(__file__).parent / file_name
    with open(file_path) as f:
        raw: dict[str, dict[str, dict[str, str]] | dict[str, str] | list[str]] = yaml.safe_load(f)
        dumped: EagerValue = _as_config_ctor(WasmTemplatesConfig)(**raw).model_dump()
        assert _is_wasm_templates_dict(dumped)
        return dumped


def load_yaml_dir(dir_name: str) -> dict[str, dict[str, dict[str, str]] | dict[str, str] | list[str]]:
    """Load all yaml files in a directory.

    Args:
        dir_name: Input `dir_name` argument (dir name).

    Returns:
        dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]]: Computed result.
    """
    import glob

    dir_path: Path = Path(__file__).parent / dir_name
    res = {}
    if dir_path.is_dir():
        for filename in glob.glob(str(dir_path / "*.yaml")):
            with open(filename) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                source = data.get("templates", data)
                for k, v in source.items():
                    if isinstance(v, dict) and "body" in v:
                        res[k] = v
                    elif isinstance(v, str):
                        res[k] = {"body": v}
                    else:
                        res[k] = v
    return {"templates": res}


def get_wasm_template(template_name: str) -> dict[str, str]:
    """Get template.

    Args:
        template_name (str): Name of the template.

    Returns:
        dict[str, str]: Template dictionary mapping string keys to string values.
    """
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    templates: dict[str, dict[str, str]] | dict[str, str] | list[str] = _WASM_TEMPLATES.get("templates", {})
    if isinstance(templates, dict):
        res = templates.get(template_name)
        if not res:
            res = next((v for k, v in templates.items() if k.lower() == template_name.lower()), {})
        if isinstance(res, dict):
            return {str(k): str(v) for k, v in res.items()}
    return {}


def get_js_orchestration_template(name: str) -> str:
    """Get js template.

    Args:
        name: Input `name` argument (name).

    Returns:
        str: Computed result.
    """
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    js_orch: dict[str, dict[str, str]] | dict[str, str] | list[str] = _WASM_TEMPLATES.get("js_orchestration", {})
    if isinstance(js_orch, dict):
        return str(js_orch.get(name, ""))
    return ""


def get_cpp_helpers() -> list[str]:
    """Get cpp helpers.

    Returns:
        list[str]: Computed result.
    """
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    helpers = _WASM_TEMPLATES.get("cpp_helpers", [])
    if isinstance(helpers, list):
        return [str(h) for h in helpers]
    return []


_WASM_SIMD_OPS: dict[str, EagerValue] = {}


def get_wasm_simd_ops() -> dict[str, EagerValue]:
    """Retrieve full WASM SIMD operation mappings.

    Returns:
        dict[str, object]: Raw or parsed WASM SIMD operations dictionary.
    """
    global _WASM_SIMD_OPS
    if not _WASM_SIMD_OPS:
        simd_path: Path = Path(__file__).parent / "wasm_simd_ops.yaml"
        if simd_path.is_file():
            with open(simd_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                _WASM_SIMD_OPS = data
    return _WASM_SIMD_OPS


def get_wasm_simd_op(op_name: str) -> dict[str, str | int | float | None]:
    """Retrieve SIMD intrinsic specification for an operation.

    Args:
        op_name (str): Operation name.

    Returns:
        dict[str, Union[str, int, float, None]]: Op SIMD configuration or empty dict.
    """
    ops = get_wasm_simd_ops().get("operations", {})
    if isinstance(ops, dict) and op_name in ops:
        val = ops[op_name]
        if isinstance(val, dict):
            return val
    return {}
