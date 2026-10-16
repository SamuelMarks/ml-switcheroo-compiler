"""Standardize all operation YAML definitions with declarative schemas."""

from __future__ import annotations

import os
from typing import Union

import yaml

from ml_switcheroo_compiler.ops.config_models import (
    OperationDefinitionModel,
)
from ml_switcheroo_compiler.ops.shape_inference import load_shape_signatures

OpYamlValue = Union[
    str,
    int,
    float,
    bool,
    list["OpYamlValue"],
    dict[str, "OpYamlValue"],
    None,
]
OpYamlDict = dict[str, OpYamlValue]

DEFS_DIR: str = "src/ml_switcheroo_compiler/ops/definitions"

COMMUTATIVE_OPS: set[str] = {
    "Add",
    "Mul",
    "Multiply",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Equal",
    "NotEqual",
    "Maximum",
    "Minimum",
}

ASSOCIATIVE_OPS: set[str] = {
    "Add",
    "Mul",
    "Multiply",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Maximum",
    "Minimum",
}

IDEMPOTENT_OPS: set[str] = {
    "Relu",
    "Floor",
    "Ceil",
    "Round",
    "Abs",
}

IDENTITY_VALUES: dict[str, int | float] = {
    "Add": 0,
    "Sub": 0,
    "Mul": 1,
    "Multiply": 1,
    "BitwiseOr": 0,
    "BitwiseXor": 0,
}

ABSORBING_VALUES: dict[str, int | float] = {
    "Mul": 0,
    "Multiply": 0,
    "BitwiseAnd": 0,
}


def build_signature_section(
    op_name: str,
    raw_data: OpYamlDict,
) -> OpYamlDict:
    """Build standardized declarative signature section for an operation.

    Args:
        op_name (str): The canonical name of the operation.
        raw_data (OpYamlDict): Raw dictionary data of the op.

    Returns:
        OpYamlDict: Validated signature section dictionary.
    """
    raw_std_args = raw_data.get("std_args")
    args_list: list[OpYamlValue] = []
    input_tensors: list[OpYamlValue] = []
    has_variadic: bool = False

    if isinstance(raw_std_args, list):
        for item in raw_std_args:
            if isinstance(item, dict):
                arg_name = str(item.get("name", "x"))
                arg_type = str(item.get("type", "array"))
                is_var = bool(item.get("is_variadic", False))
                raw_kind = item.get("kind", "positional_or_keyword")
                kind = str(raw_kind) if raw_kind is not None else None
                if is_var:
                    has_variadic = True
                if "array" in arg_type or "tensor" in arg_type:
                    input_tensors.append(arg_name)
                args_list.append(
                    {
                        "name": arg_name,
                        "type": arg_type,
                        "is_variadic": is_var,
                        "kind": kind,
                    }
                )
    if not input_tensors:
        input_tensors = ["x"] if not args_list else [str(a["name"]) for a in args_list if isinstance(a, dict)]

    sig_dict: OpYamlDict = {
        "args": args_list,
        "input_tensors": input_tensors,
        "output_tensors": ["out"],
        "has_variadic_args": has_variadic,
        "has_variadic_kwargs": False,
        "keyword_constraints": {},
        "keyword_defaults": {},
    }
    return sig_dict


def build_dtype_rules_section(
    op_name: str,
    raw_data: OpYamlDict,
) -> OpYamlDict:
    """Build standardized declarative dtype_rules section.

    Args:
        op_name (str): The canonical name of the operation.
        raw_data (OpYamlDict): Raw dictionary data of the op.

    Returns:
        OpYamlDict: Validated dtype_rules section dictionary.
    """
    if op_name in {"Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual"}:
        out_inference = "bool"
        promo = "boolean"
    elif op_name in {"BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot"}:
        out_inference = "in0"
        promo = "integer"
    elif "Cast" in op_name:
        out_inference = "target_dtype"
        promo = "cast"
    else:
        std_args = raw_data.get("std_args")
        num_args = len(std_args) if isinstance(std_args, list) else 0
        out_inference = "promote(in0, in1)" if num_args > 1 else "in0"
        promo = "standard"

    allowed: list[OpYamlValue] = ["all"]
    return {
        "promotion_matrix": promo,
        "allowed_input_dtypes": allowed,
        "output_dtype_inference": out_inference,
    }


def build_shape_signature_section(
    op_name: str,
    known_signatures: dict[str, dict[str, str]],
) -> OpYamlDict:
    """Build standardized declarative shape_signature section.

    Args:
        op_name (str): The canonical name of the operation.
        known_signatures (dict[str, dict[str, str]]): Loaded signatures from shape_signatures.yaml.

    Returns:
        OpYamlDict: Validated shape_signature section dictionary.
    """
    if op_name in known_signatures:
        entry = known_signatures[op_name]
        pattern = entry.get("pattern", "broadcast(in0, in1)")
        cat = entry.get("category", "elementwise")
    elif op_name in {"MatMul", "BatchMatMul", "Bmm"}:
        pattern = "matmul(in0, in1)"
        cat = "linalg"
    elif op_name in {"Conv1D", "Conv2D", "Conv3D"}:
        pattern = "conv(in0, in1)"
        cat = "nn"
    elif op_name in COMMUTATIVE_OPS:
        pattern = "broadcast(in0, in1)"
        cat = "elementwise"
    else:
        pattern = "in0"
        cat = "unary"

    sym_expr = "[*shape], [*shape] -> [*shape]" if "broadcast" in pattern else "[*shape] -> [*shape]"
    if "matmul" in pattern:
        sym_expr = "[B, M, K], [B, K, N] -> [B, M, N]"

    return {
        "pattern": pattern,
        "symbolic_expression": sym_expr,
        "category": cat,
    }


def build_invariants_section(op_name: str) -> OpYamlDict:
    """Build standardized declarative invariants section.

    Args:
        op_name (str): The canonical name of the operation.

    Returns:
        OpYamlDict: Validated invariants section dictionary.
    """
    return {
        "is_commutative": op_name in COMMUTATIVE_OPS,
        "is_associative": op_name in ASSOCIATIVE_OPS,
        "is_idempotent": op_name in IDEMPOTENT_OPS,
        "identity_value": IDENTITY_VALUES.get(op_name),
        "absorbing_value": ABSORBING_VALUES.get(op_name),
    }


def build_autodiff_section(
    op_name: str,
    raw_data: OpYamlDict,
) -> OpYamlDict:
    """Build standardized declarative autodiff section.

    Args:
        op_name (str): The canonical name of the operation.
        raw_data (OpYamlDict): Raw dictionary data of the op.

    Returns:
        OpYamlDict: Validated autodiff section dictionary.
    """
    raw_ad = raw_data.get("autodiff")
    vjp_val: OpYamlValue = None
    jvp_val: OpYamlValue = None

    if isinstance(raw_ad, dict):
        vjp_val = raw_ad.get("vjp")
        jvp_val = raw_ad.get("jvp")
    elif "vjp" in raw_data or "jvp" in raw_data:
        vjp_val = raw_data.get("vjp")
        jvp_val = raw_data.get("jvp")

    return {
        "vjp_rule_id": f"{op_name}_vjp",
        "jvp_rule_id": f"{op_name}_jvp",
        "vjp": vjp_val,
        "jvp": jvp_val,
    }


def standardize_op_data(
    op_name: str,
    data: OpYamlDict,
    known_signatures: dict[str, dict[str, str]],
) -> OpYamlDict:
    """Standardize single operation data dictionary with all required schema sections.

    Args:
        op_name (str): Operation identifier.
        data (OpYamlDict): Raw operation dictionary.
        known_signatures (dict[str, dict[str, str]]): Known shape signatures.

    Returns:
        OpYamlDict: Enriched and validated operation dictionary.
    """
    res: OpYamlDict = dict(data)
    res["signature"] = build_signature_section(op_name, res)
    res["dtype_rules"] = build_dtype_rules_section(op_name, res)
    res["shape_signature"] = build_shape_signature_section(op_name, known_signatures)
    res["invariants"] = build_invariants_section(op_name)
    res["autodiff"] = build_autodiff_section(op_name, res)

    # Validate against Pydantic model
    OperationDefinitionModel.model_validate(res)
    return res


def run_standardization() -> int:
    """Run standardization across all YAML files in definitions directory.

    Returns:
        int: Total number of standardized files.
    """
    shape_sig_cfg = load_shape_signatures()
    known_sigs: dict[str, dict[str, str]] = {name: {"pattern": op.pattern or "in0", "category": op.category or "elementwise"} for name, op in shape_sig_cfg.operations.items()}

    files = sorted(f for f in os.listdir(DEFS_DIR) if f.endswith(".yaml"))
    count = 0

    for filename in files:
        filepath = os.path.join(DEFS_DIR, filename)
        with open(filepath, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        if not isinstance(content, dict):
            continue

        if "operation" in content:
            op_name = str(content["operation"])
            standardized = standardize_op_data(op_name, content, known_sigs)
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(standardized, f, sort_keys=False, indent=2)
            count += 1
        else:
            # Multi-op domain registry file
            multi_res: dict[str, OpYamlDict] = {}
            for op_inner, data_inner in content.items():
                if isinstance(data_inner, dict):
                    multi_res[op_inner] = standardize_op_data(op_inner, data_inner, known_sigs)
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(multi_res, f, sort_keys=False, indent=2)
            count += 1

    return count


def main() -> None:
    """Execute standardization workflow.

    Returns:
        None
    """
    total = run_standardization()
    print(f"Successfully standardized {total} operation YAML definition files.")


if __name__ == "__main__":
    main()
