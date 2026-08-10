"""Build script for the operation registry."""

import os
import pprint

import yaml


def build() -> None:
    """Build the python registry from YAML definitions."""
    definitions_dir = "src/ml_switcheroo_compiler/ops/definitions"
    out_file = "src/ml_switcheroo_compiler/ops/generated_registry.py"

    ops_data = {}

    # Load all yaml files
    for filename in sorted(os.listdir(definitions_dir)):
        if filename.endswith(".yaml"):
            with open(os.path.join(definitions_dir, filename)) as f:
                data = yaml.safe_load(f)
                if "operation" in data:
                    op_name = data["operation"]
                    ops_data[op_name] = data
                else:
                    for op_name, op_info in data.items():
                        ops_data[op_name] = op_info

    # Generate the python file
    with open(out_file, "w") as f:
        f.write('"""Auto-generated operation registry."""\n\n')
        f.write("# AUTO-GENERATED FILE. DO NOT EDIT.\n")
        f.write("# Generated from src/ml_switcheroo_compiler/ops/definitions/*.yaml\n\n")

        # __all__ definition
        all_list_str = ',\n    "OPS_REGISTRY"'
        f.write(f"__all__ = [\n    {all_list_str.strip(', ')}\n]\n\n")

        # Format the dictionary directly into python source
        formatted_dict = pprint.pformat(ops_data, indent=4, sort_dicts=True)
        f.write(f"OPS_REGISTRY = {formatted_dict}\n")


if __name__ == "__main__":
    build()
