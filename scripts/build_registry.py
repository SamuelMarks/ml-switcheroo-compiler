"""Build script for the operation registry."""

from __future__ import annotations

import os
import pprint

import yaml

from ml_switcheroo_compiler.ops.generated_registry import (
    OpRegistryValue,
    _normalize_and_validate_op,
    load_backend_mappings,
)


def build(
    definitions_dir: str = "src/ml_switcheroo_compiler/ops/definitions",
    out_file: str = "src/ml_switcheroo_compiler/ops/generated_registry.py",
    out_yaml: str | None = None,
    backends_dir: str = "src/ml_switcheroo_compiler/backends",
) -> None:
    """Build the python and yaml registry from declarative definitions.

    Args:
        definitions_dir (str): Path to directory containing YAML operation definitions.
        out_file (str): Path to destination python file.
        out_yaml (str | None): Path to destination YAML registry file.
        backends_dir (str): Path to backends directory containing mapping files.

    Returns:
        None
    """
    backend_mappings = load_backend_mappings(backends_dir)
    ops_data: dict[str, dict[str, OpRegistryValue]] = {}

    # Load all yaml files
    for filename in sorted(os.listdir(definitions_dir)):
        if filename.endswith(".yaml"):
            with open(os.path.join(definitions_dir, filename), encoding="utf-8") as f:
                data: dict[str, OpRegistryValue] = yaml.safe_load(f)
                if isinstance(data, dict):
                    if "operation" in data:
                        op_name: str = str(data["operation"])
                        normalized_op = _normalize_and_validate_op(op_name, data, backend_mappings)
                        ops_data[op_name] = normalized_op
                        if "aliases" in data and isinstance(data["aliases"], list):
                            for alias in data["aliases"]:
                                ops_data[str(alias)] = normalized_op
                    else:
                        for op_name_inner, op_info in data.items():
                            if isinstance(op_info, dict):
                                normalized_op = _normalize_and_validate_op(op_name_inner, op_info, backend_mappings)
                                ops_data[op_name_inner] = normalized_op

    # Generate the python file
    with open(out_file, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated operation registry."""\n\n')
        f.write("# AUTO-GENERATED FILE. DO NOT EDIT.\n")
        f.write("# Generated from src/ml_switcheroo_compiler/ops/definitions/*.yaml\n\nfrom __future__ import annotations\n\nimport typing\nfrom collections.abc import Mapping\n\n")

        # __all__ definition
        all_list_str: str = ',\n    "OPS_REGISTRY"'
        f.write(f"__all__ = [\n    {all_list_str.strip(', ')}\n]\n\n")
        f.write('OpRegistryValue = typing.Union[\n    str,\n    int,\n    float,\n    bool,\n    list["OpRegistryValue"],\n    dict[str, "OpRegistryValue"],\n    Mapping[str, "OpRegistryValue"],\n    tuple["OpRegistryValue", ...],\n    None,\n]\n\n')

        # Format the dictionary directly into python source
        formatted_dict: str = pprint.pformat(ops_data, indent=4, sort_dicts=True)
        f.write(f"OPS_REGISTRY: dict[str, dict[str, OpRegistryValue]] = {formatted_dict}\n")

    # Generate the yaml file only if explicitly specified
    if out_yaml:
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(ops_data, f, sort_keys=True, indent=2)


def main() -> None:
    """Entry point for the script.

    Returns:
        None
    """
    build()


if __name__ == "__main__":
    main()
