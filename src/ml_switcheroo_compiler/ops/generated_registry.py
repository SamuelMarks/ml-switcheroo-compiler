"""Operation registry loader synthesized from declarative YAML definitions."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Mapping

import yaml

from ml_switcheroo_compiler.ops.config_models import OperationDefinitionModel

CPP_SCALAR_EXPR: dict[str, str] = {
    "Abs": "std::abs(in0_val)",
    "Acos": "std::acos(in0_val)",
    "Acosh": "std::acosh(in0_val)",
    "Add": "in0_val + in1_val",
    "Asin": "std::asin(in0_val)",
    "Asinh": "std::asinh(in0_val)",
    "Atan": "std::atan(in0_val)",
    "Atan2": "std::atan2(in0_val, in1_val)",
    "Atanh": "std::atanh(in0_val)",
    "BitwiseAnd": "(int64_t)in0_val & (int64_t)in1_val",
    "BitwiseOr": "(int64_t)in0_val | (int64_t)in1_val",
    "BitwiseXor": "(int64_t)in0_val ^ (int64_t)in1_val",
    "Broadcast": "in0_val",
    "BroadcastTo": "in0_val",
    "Cast": "(float)in0_val",
    "Cbrt": "std::cbrt(in0_val)",
    "Ceil": "std::ceil(in0_val)",
    "Clone": "in0_val",
    "Copy": "in0_val",
    "Cos": "std::cos(in0_val)",
    "Cosh": "std::cosh(in0_val)",
    "Div": "in0_val / in1_val",
    "Divide": "in0_val / in1_val",
    "Elu": "in0_val > 0 ? in0_val : std::exp(in0_val) - 1.0f",
    "Equal": "in0_val == in1_val ? 1.0f : 0.0f",
    "Erf": "std::erf(in0_val)",
    "Erfc": "std::erfc(in0_val)",
    "Exp": "std::exp(in0_val)",
    "Exp2": "std::exp2(in0_val)",
    "Expm1": "std::expm1(in0_val)",
    "Floor": "std::floor(in0_val)",
    "FloorDiv": "std::floor(in0_val / in1_val)",
    "FloorDivide": "std::floor(in0_val / in1_val)",
    "Fmax": "std::fmax(in0_val, in1_val)",
    "Fmin": "std::fmin(in0_val, in1_val)",
    "Fmod": "std::fmod(in0_val, in1_val)",
    "Gelu": "0.5f * in0_val * (1.0f + std::erf(in0_val * 0.70710678f))",
    "Greater": "in0_val > in1_val ? 1.0f : 0.0f",
    "GreaterEqual": "in0_val >= in1_val ? 1.0f : 0.0f",
    "HardSigmoid": "std::max(0.0f, std::min(1.0f, 0.2f * in0_val + 0.5f))",
    "Hypot": "std::hypot(in0_val, in1_val)",
    "Identity": "in0_val",
    "LeakyRelu": "in0_val > 0 ? in0_val : 0.01f * in0_val",
    "Less": "in0_val < in1_val ? 1.0f : 0.0f",
    "LessEqual": "in0_val <= in1_val ? 1.0f : 0.0f",
    "Log": "std::log(in0_val)",
    "Log10": "std::log10(in0_val)",
    "Log1p": "std::log1p(in0_val)",
    "Log2": "std::log2(in0_val)",
    "Maximum": "std::max(in0_val, in1_val)",
    "Minimum": "std::min(in0_val, in1_val)",
    "Mish": "in0_val * std::tanh(std::log1p(std::exp(in0_val)))",
    "Mod": "std::fmod(in0_val, in1_val)",
    "Mul": "in0_val * in1_val",
    "Multiply": "in0_val * in1_val",
    "Neg": "-in0_val",
    "Negative": "-in0_val",
    "NotEqual": "in0_val != in1_val ? 1.0f : 0.0f",
    "Pow": "std::pow(in0_val, in1_val)",
    "Relu": "std::max(0.0f, in0_val)",
    "Remainder": "std::remainder(in0_val, in1_val)",
    "Rint": "std::rint(in0_val)",
    "Round": "std::round(in0_val)",
    "Rsqrt": "1.0f / std::sqrt(in0_val)",
    "Selu": "1.0507f * (in0_val > 0 ? in0_val : 1.67326f * (std::exp(in0_val) - 1.0f))",
    "Sigmoid": "1.0f / (1.0f + std::exp(-in0_val))",
    "Sign": "(in0_val > 0.0f) - (in0_val < 0.0f)",
    "Silu": "in0_val / (1.0f + std::exp(-in0_val))",
    "Sin": "std::sin(in0_val)",
    "Sinh": "std::sinh(in0_val)",
    "Softplus": "std::log1p(std::exp(in0_val))",
    "Sqrt": "std::sqrt(in0_val)",
    "Sub": "in0_val - in1_val",
    "Subtract": "in0_val - in1_val",
    "Swish": "in0_val / (1.0f + std::exp(-in0_val))",
    "Tan": "std::tan(in0_val)",
    "Tanh": "std::tanh(in0_val)",
    "TrueDivide": "in0_val / in1_val",
    "Trunc": "std::trunc(in0_val)",
    "TruncateDiv": "(float)((int64_t)in0_val / (int64_t)in1_val)",
    "TruncateMod": "(float)((int64_t)in0_val % (int64_t)in1_val)",
}


_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_backend_mappings(backends_dir: str = "src/ml_switcheroo_compiler/backends") -> dict[str, dict[str, str]]:
    """Load backend mappings from the compiler backends directory.

    Args:
        backends_dir (str): Root backends directory path.

    Returns:
        dict[str, dict[str, str]]: Mappings dictionary per backend and op.
    """
    mappings: dict[str, dict[str, str]] = {}
    if not os.path.isdir(backends_dir):
        return mappings

    for b in sorted(os.listdir(backends_dir)):
        map_dir = os.path.join(backends_dir, b, "mappings")
        if os.path.isdir(map_dir):
            mappings[b] = {}
            for y in sorted(os.listdir(map_dir)):
                if y.endswith(".yaml"):
                    try:
                        with open(os.path.join(map_dir, y), encoding="utf-8") as fp:
                            d = yaml.load(fp, Loader=_YAML_LOADER)
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if isinstance(v, dict) and "target_api" in v:
                                    mappings[b][str(k)] = str(v["target_api"])
                    except (OSError, yaml.YAMLError):
                        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return mappings


OpRegistryValue = typing.Union[str, int, float, bool, list["OpRegistryValue"], dict[str, "OpRegistryValue"], Mapping[str, "OpRegistryValue"], tuple["OpRegistryValue", ...], "None"]


def _normalize_backend_variant_rule(
    op_name: str,
    backend: str,
    rule: OpRegistryValue,
    backend_mappings: dict[str, dict[str, str]],
) -> dict[str, str | int | float | bool | None]:
    """Normalize a single backend variant dictionary for an operation.

    Args:
        op_name: Canonical operation name.
        backend: Target backend identifier.
        rule: Raw variant rule specification.
        backend_mappings: Loaded backend mappings per target framework.

    Returns:
        Normalized backend variant rule dictionary.
    """
    if not isinstance(rule, dict) or len(rule) == 0:
        if backend in backend_mappings and op_name in backend_mappings[backend]:
            return {
                "target_api": backend_mappings[backend][op_name],
                "supported": True,
            }
        return {"supported": False}

    rule_dict = dict(rule)
    if backend == "llvm_cpp" and rule_dict.get("scalar_expr") == "in0_val" and op_name.lower() not in ("identity", "copy", "clone", "passthrough"):
        if op_name in CPP_SCALAR_EXPR:
            rule_dict["scalar_expr"] = CPP_SCALAR_EXPR[op_name]
            rule_dict["supported"] = True
            return rule_dict
        return {"supported": False}

    rule_dict.setdefault("supported", True)
    return rule_dict


def _normalize_and_validate_op(
    op_name: str,
    op_data: dict[str, OpRegistryValue],
    backend_mappings: dict[str, dict[str, str]],
    validate: bool = True,
) -> dict[str, OpRegistryValue]:
    """Audit, normalize, and validate operation entry, purging untyped/empty variant dictionaries.

    Args:
        op_name (str): The canonical name of the operation.
        op_data (dict[str, OpRegistryValue]): Raw dictionary data for the operation.
        backend_mappings (dict[str, dict[str, str]]): Loaded backend mappings per target framework.
        validate (bool): Whether to run Pydantic schema validation on the normalized dictionary.

    Returns:
        dict[str, OpRegistryValue]: Normalized and validated operation dictionary.
    """
    data: dict[str, OpRegistryValue] = dict(op_data)
    variants = data.get("variants")
    if isinstance(variants, dict):
        data["variants"] = {backend: _normalize_backend_variant_rule(op_name, backend, rule, backend_mappings) for backend, rule in variants.items()}

    if "math_semantics" not in data:
        data["math_semantics"] = {
            "is_commutative": op_name in {"Add", "Mul", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Equal", "NotEqual"},
            "is_associative": op_name in {"Add", "Mul", "BitwiseAnd", "BitwiseOr", "BitwiseXor"},
            "is_idempotent": op_name in {"Relu", "Floor", "Ceil", "Round", "Abs"},
            "differentiable": True,
        }

    if validate:
        OperationDefinitionModel.model_validate(data)
    return data


def _build_ops_registry() -> dict[str, dict[str, OpRegistryValue]]:
    """Load and normalize the operation registry from declarative YAML files.

    Returns:
        dict[str, dict[str, OpRegistryValue]]: Normalized operation registry dictionary.
    """
    defs_dir = os.path.join(os.path.dirname(__file__), "definitions")
    backends_dir = os.path.join(os.path.dirname(__file__), "..", "backends")
    backend_mappings = load_backend_mappings(backends_dir)
    ops_data: dict[str, dict[str, OpRegistryValue]] = {}

    for filename in sorted(f for f in os.listdir(defs_dir) if f.endswith(".yaml")):
        with open(os.path.join(defs_dir, filename), encoding="utf-8") as f:
            data = yaml.load(f, Loader=_YAML_LOADER)
        if "operation" in data:
            op_name = str(data["operation"])
            normalized_op = _normalize_and_validate_op(op_name, data, backend_mappings, validate=False)
            ops_data[op_name] = normalized_op
            if "aliases" in data and isinstance(data["aliases"], list):
                for alias in data["aliases"]:
                    ops_data[str(alias)] = normalized_op
        else:
            for op_name_inner, op_info in data.items():
                normalized_op = _normalize_and_validate_op(op_name_inner, op_info, backend_mappings, validate=False)
                ops_data[op_name_inner] = normalized_op
    return ops_data


OPS_REGISTRY: dict[str, dict[str, OpRegistryValue]] = _build_ops_registry()

__all__ = ["CPP_SCALAR_EXPR", "OPS_REGISTRY", "_normalize_and_validate_op", "load_backend_mappings"]
