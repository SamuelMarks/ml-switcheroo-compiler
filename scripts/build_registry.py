"""Build script for the operation registry."""

import os
import pprint
import typing

import yaml


def build(definitions_dir: str = "src/ml_switcheroo_compiler/ops/definitions", out_file: str = "src/ml_switcheroo_compiler/ops/generated_registry.py") -> None:
    """Build the python registry from YAML definitions.

    Args:
        definitions_dir: Path to the directory containing YAML files.
        out_file: Path to the output python file.

    Returns:
        None
    """
    ops_data: dict[str, dict[str, typing.Union[str, int, float, bool, list, dict, tuple, None]]] = {}

    # Load all yaml files
    for filename in sorted(os.listdir(definitions_dir)):
        if filename.endswith(".yaml"):
            with open(os.path.join(definitions_dir, filename)) as f:
                data: dict[str, typing.Union[str, int, float, bool, list, dict, tuple, None]] = yaml.safe_load(f)
                if "operation" in data:
                    op_name: str = str(data["operation"])
                    ops_data[op_name] = data
                else:
                    for op_name_inner, op_info in data.items():
                        if isinstance(op_info, dict):
                            ops_data[op_name_inner] = op_info

    # Generate the python file
    with open(out_file, "w") as f:
        f.write('"""Auto-generated operation registry."""\n\n')
        f.write("# AUTO-GENERATED FILE. DO NOT EDIT.\n")
        f.write("# Generated from src/ml_switcheroo_compiler/ops/definitions/*.yaml\n\nimport typing\n\n")

        # __all__ definition
        all_list_str: str = ',\n    "OPS_REGISTRY"'
        f.write(f"__all__ = [\n    {all_list_str.strip(', ')}\n]\n\n")

        # Format the dictionary directly into python source
        formatted_dict: str = pprint.pformat(ops_data, indent=4, sort_dicts=True)
        f.write(f"OPS_REGISTRY: dict[str, dict[str, typing.Union[str, int, float, bool, list, dict, tuple, None]]] = {formatted_dict}\n")


def main() -> None:
    """Entry point for the script.

    Returns:
        None
    """
    build()


if __name__ == "__main__":
    main()
