"""Standardize all backend mapping files to the unified declarative schema."""

from __future__ import annotations

import glob
import os

import yaml

from ml_switcheroo_compiler.backends.mapping_loader import OpMappingSchema

BACKENDS_DIR: str = "src/ml_switcheroo_compiler/backends"


def build_default_ast_template(target_api: str, op_name: str) -> str:
    """Generate a clean AST template string from target_api and operation characteristics.

    Args:
        target_api (str): Target API endpoint.
        op_name (str): Operation identifier.

    Returns:
        str: Standardized AST format template string.
    """
    if not target_api or target_api == "custom_op":
        return "{out} = {in0}"

    unary_ops = {"Abs", "Acos", "Asin", "Atan", "Ceil", "Cos", "Cosh", "Exp", "Floor", "Log", "Neg", "Relu", "Sin", "Sinh", "Sqrt", "Tan", "Tanh"}
    if op_name in unary_ops:
        return f"{{out}} = {target_api}({{in0}})"
    binary_ops = {"Add", "Sub", "Mul", "Div", "FloorDiv", "Mod", "Pow", "Matmul", "Atan2", "Equal", "NotEqual", "Greater", "Less"}
    if op_name in binary_ops:
        return f"{{out}} = {target_api}({{in0}}, {{in1}})"
    return f"{{out}} = {target_api}({{inputs}})"


def standardize_mapping_data(
    backend: str,
    op_name: str,
    raw_spec: dict[str, object],
) -> dict[str, object]:
    """Standardize single operation mapping dictionary to unified schema.

    Args:
        backend (str): Backend framework name.
        op_name (str): Operation name.
        raw_spec (dict[str, object]): Existing specification dictionary.

    Returns:
        dict[str, object]: Standardized declarative mapping dictionary.
    """
    target_api = str(raw_spec.get("target_api") or "")
    existing_template = str(raw_spec.get("ast_template") or "")
    if existing_template:
        ast_template = existing_template
    else:
        ast_template = build_default_ast_template(target_api, op_name)

    kwarg_map: dict[str, str | None] = {}
    raw_kwarg_map = raw_spec.get("kwarg_map") or raw_spec.get("kwarg_translations") or {}
    if isinstance(raw_kwarg_map, dict):
        for k, v in raw_kwarg_map.items():
            kwarg_map[str(k)] = str(v) if v is not None else None

    dtype_overrides: dict[str, str] = {}
    raw_dtype_overrides = raw_spec.get("dtype_overrides") or {}
    if isinstance(raw_dtype_overrides, dict):
        for k, v in raw_dtype_overrides.items():
            dtype_overrides[str(k)] = str(v)

    res: dict[str, object] = {
        "operation": op_name,
        "backend": backend,
        "target_api": target_api,
        "ast_template": ast_template,
        "kwarg_map": kwarg_map,
        "dtype_overrides": dtype_overrides,
    }

    # Validate against OpMappingSchema
    OpMappingSchema.model_validate(res)
    return res


def run_standardization() -> int:
    """Standardize all mapping files across all backends.

    Returns:
        int: Number of files processed.
    """
    pattern = os.path.join(BACKENDS_DIR, "*/mappings/*.yaml")
    files = sorted(glob.glob(pattern))
    count = 0

    for filepath in files:
        parts = filepath.split(os.sep)
        backend = parts[-3]
        filename = os.path.splitext(parts[-1])[0]

        with open(filepath, encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}

        if not isinstance(content, dict):
            continue

        if "operation" in content and isinstance(content["operation"], str):
            op_name = content["operation"]
            standardized = standardize_mapping_data(backend, op_name, content)
        elif filename in content and isinstance(content[filename], dict):
            op_name = filename
            standardized = standardize_mapping_data(backend, op_name, content[filename])
        else:
            first_key = next(iter(content.keys()), filename)
            spec = content.get(first_key) if isinstance(content.get(first_key), dict) else content
            op_name = str(first_key)
            standardized = standardize_mapping_data(backend, op_name, spec)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(standardized, f, sort_keys=False, indent=2)
        count += 1

    return count


def main() -> None:
    """Execute standardization workflow.

    Returns:
        None
    """
    total = run_standardization()
    print(f"Successfully standardized {total} backend mapping YAML files.")


if __name__ == "__main__":
    main()
