"""Generate and migrate declarative domain-specific operation YAML files."""

from __future__ import annotations

import os
from typing import Any

import yaml

from ml_switcheroo_compiler.ops.config_models import DomainOperationsModel, OperationDefinitionModel

BASE_DIR: str = "src/ml_switcheroo_compiler/ops/definitions"

# Op name canonical aliases mapping
NAME_ALIASES: dict[str, str] = {
    "FloorDiv": "FloorDivide",
    "NormL1": "Norm",
    "NormL2": "Norm",
    "BatchMatMul": "Bmm",
    "VectorDot": "Vdot",
    "MatrixInverse": "Inv",
    "Determinant": "Det",
    "RandomNormal": "Normal",
    "RandomUniform": "Uniform",
    "Categorical": "Categorical",
    "StandardGamma": "Gamma",
    "Exponential": "Exponential",
    "Poisson": "Poisson",
    "Map": "ShardMapFn",
    "ResizeBilinear": "Resize",
    "ResizeNearest": "Resize",
}

CPP_SCALAR_EXPR: dict[str, str] = {
    "Add": "in0_val + in1_val",
    "Sub": "in0_val - in1_val",
    "Mul": "in0_val * in1_val",
    "Div": "in0_val / in1_val",
    "FloorDiv": "std::floor(in0_val / in1_val)",
    "FloorDivide": "std::floor(in0_val / in1_val)",
    "Mod": "std::fmod(in0_val, in1_val)",
    "Pow": "std::pow(in0_val, in1_val)",
    "BitwiseAnd": "(int64_t)in0_val & (int64_t)in1_val",
    "BitwiseOr": "(int64_t)in0_val | (int64_t)in1_val",
    "BitwiseXor": "(int64_t)in0_val ^ (int64_t)in1_val",
    "Equal": "in0_val == in1_val",
    "NotEqual": "in0_val != in1_val",
    "Greater": "in0_val > in1_val",
    "Less": "in0_val < in1_val",
    "GreaterEqual": "in0_val >= in1_val",
    "LessEqual": "in0_val <= in1_val",
    "Abs": "std::abs(in0_val)",
    "Neg": "-in0_val",
    "Sign": "(in0_val > 0) - (in0_val < 0)",
    "Exp": "std::exp(in0_val)",
    "Log": "std::log(in0_val)",
    "Log2": "std::log2(in0_val)",
    "Log10": "std::log10(in0_val)",
    "Sqrt": "std::sqrt(in0_val)",
    "Rsqrt": "1.0f / std::sqrt(in0_val)",
    "Sin": "std::sin(in0_val)",
    "Cos": "std::cos(in0_val)",
    "Tan": "std::tan(in0_val)",
    "Sinh": "std::sinh(in0_val)",
    "Cosh": "std::cosh(in0_val)",
    "Tanh": "std::tanh(in0_val)",
    "Asin": "std::asin(in0_val)",
    "Acos": "std::acos(in0_val)",
    "Atan": "std::atan(in0_val)",
    "Floor": "std::floor(in0_val)",
    "Ceil": "std::ceil(in0_val)",
    "Round": "std::round(in0_val)",
    "Erf": "std::erf(in0_val)",
    "Relu": "std::max(0.0f, in0_val)",
    "Gelu": "0.5f * in0_val * (1.0f + std::erf(in0_val * 0.70710678f))",
    "Silu": "in0_val / (1.0f + std::exp(-in0_val))",
    "Elu": "in0_val > 0 ? in0_val : std::exp(in0_val) - 1.0f",
    "Selu": "1.0507f * (in0_val > 0 ? in0_val : 1.67326f * (std::exp(in0_val) - 1.0f))",
    "LeakyRelu": "in0_val > 0 ? in0_val : 0.01f * in0_val",
    "Softplus": "std::log1p(std::exp(in0_val))",
    "Sigmoid": "1.0f / (1.0f + std::exp(-in0_val))",
    "HardSigmoid": "std::max(0.0f, std::min(1.0f, 0.2f * in0_val + 0.5f))",
    "Mish": "in0_val * std::tanh(std::log1p(std::exp(in0_val)))",
    "Swish": "in0_val / (1.0f + std::exp(-in0_val))",
}

DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "binary_ops.yaml": {
        "domain": "binary",
        "ops": [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "FloorDiv",
            "Mod",
            "Pow",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "Equal",
            "NotEqual",
            "Greater",
            "Less",
            "GreaterEqual",
            "LessEqual",
        ],
        "inputs": ["x1", "x2"],
        "is_binary": True,
    },
    "unary_ops.yaml": {
        "domain": "unary",
        "ops": [
            "Abs",
            "Neg",
            "Sign",
            "Exp",
            "Log",
            "Log2",
            "Log10",
            "Sqrt",
            "Rsqrt",
            "Sin",
            "Cos",
            "Tan",
            "Sinh",
            "Cosh",
            "Tanh",
            "Asin",
            "Acos",
            "Atan",
            "Floor",
            "Ceil",
            "Round",
            "Erf",
        ],
        "inputs": ["x"],
        "is_binary": False,
    },
    "activations.yaml": {
        "domain": "activation",
        "ops": [
            "Relu",
            "Gelu",
            "Silu",
            "Elu",
            "Selu",
            "LeakyRelu",
            "Softplus",
            "Softmax",
            "LogSoftmax",
            "Sigmoid",
            "HardSigmoid",
            "Mish",
            "Swish",
        ],
        "inputs": ["x"],
        "is_binary": False,
    },
    "reductions.yaml": {
        "domain": "reduction",
        "ops": [
            "Sum",
            "Prod",
            "Mean",
            "Max",
            "Min",
            "ArgMax",
            "ArgMin",
            "All",
            "Any",
            "LogSumExp",
            "NormL1",
            "NormL2",
            "Std",
            "Var",
        ],
        "inputs": ["x"],
        "is_binary": False,
    },
    "linalg.yaml": {
        "domain": "linalg",
        "ops": [
            "MatMul",
            "BatchMatMul",
            "Dot",
            "VectorDot",
            "TensorDot",
            "Cholesky",
            "QR",
            "SVD",
            "Solve",
            "Eig",
            "MatrixInverse",
            "Determinant",
            "Trace",
        ],
        "inputs": ["a", "b"],
        "is_binary": True,
    },
    "shape_ops.yaml": {
        "domain": "shape",
        "ops": [
            "Reshape",
            "Transpose",
            "Permute",
            "Squeeze",
            "Unsqueeze",
            "ExpandDims",
            "Slice",
            "Split",
            "Concat",
            "Stack",
            "Gather",
            "Scatter",
            "Tile",
            "Pad",
            "BroadcastTo",
        ],
        "inputs": ["x"],
        "is_binary": False,
    },
    "normalization.yaml": {
        "domain": "normalization",
        "ops": ["BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm", "RMSNorm"],
        "inputs": ["x"],
        "is_binary": False,
    },
    "loss_ops.yaml": {
        "domain": "loss",
        "ops": [
            "CrossEntropyLoss",
            "MSELoss",
            "L1Loss",
            "BCELoss",
            "BCEWithLogitsLoss",
            "KLDivLoss",
            "HuberLoss",
            "NLLLoss",
            "SmoothL1Loss",
        ],
        "inputs": ["predictions", "targets"],
        "is_binary": True,
    },
    "creation_ops.yaml": {
        "domain": "creation",
        "ops": ["Zeros", "Ones", "Full", "Arange", "Linspace", "Eye", "Empty", "Meshgrid", "Tril", "Triu"],
        "inputs": [],
        "is_binary": False,
    },
    "random_ops.yaml": {
        "domain": "random",
        "ops": [
            "RandomNormal",
            "RandomUniform",
            "Bernoulli",
            "Categorical",
            "StandardGamma",
            "Exponential",
            "Poisson",
        ],
        "inputs": ["shape"],
        "is_binary": False,
    },
    "control_flow_ops.yaml": {
        "domain": "control_flow",
        "ops": ["Cond", "WhileLoop", "Scan", "Map", "Switch"],
        "inputs": ["pred", "true_fn", "false_fn"],
        "is_binary": False,
    },
    "vision_ops.yaml": {
        "domain": "vision",
        "ops": [
            "Conv1D",
            "Conv2D",
            "Conv3D",
            "ConvTranspose2D",
            "MaxPool2D",
            "AvgPool2D",
            "AdaptiveAvgPool2D",
            "ResizeBilinear",
            "ResizeNearest",
        ],
        "inputs": ["x"],
        "is_binary": False,
    },
    "signal_ops.yaml": {
        "domain": "signal",
        "ops": ["FFT", "IFFT", "RFFT", "IRFFT", "STFT", "ISTFT"],
        "inputs": ["x"],
        "is_binary": False,
    },
}


def _load_base_op(op_name: str) -> dict[str, Any]:
    """Load existing base op YAML if available.

    Args:
        op_name (str): The name of the operation.

    Returns:
        dict[str, Any]: Loaded or default op dictionary.
    """
    candidates = [op_name, NAME_ALIASES.get(op_name, ""), op_name.lower()]
    for cand in candidates:
        if not cand:
            continue
        p = os.path.join(BASE_DIR, f"{cand}.yaml")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
    return {
        "operation": op_name,
        "description": f"Declarative {op_name} operation.",
        "variants": {},
    }


def migrate_domain(filename: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Build domain-specific operations dictionary for YAML serialization.

    Args:
        filename (str): The target YAML filename.
        spec (dict[str, Any]): Domain specification configuration.

    Returns:
        dict[str, Any]: Complete validated domain operations dictionary.
    """
    domain: str = spec["domain"]
    ops: list[str] = spec["ops"]
    is_binary: bool = spec.get("is_binary", False)
    default_inputs: list[str] = spec.get("inputs", ["x"])

    domain_dict: dict[str, Any] = {}

    for op in ops:
        base_data = _load_base_op(op)
        variants = base_data.get("variants", {})

        # Enrich llvm_cpp scalar expression if known
        cpp_expr = CPP_SCALAR_EXPR.get(op) or CPP_SCALAR_EXPR.get(NAME_ALIASES.get(op, ""))
        if cpp_expr:
            if "llvm_cpp" not in variants:
                variants["llvm_cpp"] = {}
            variants["llvm_cpp"]["scalar_expr"] = cpp_expr
            variants["llvm_cpp"]["template"] = "binary" if is_binary else "unary"

        # Math semantics
        math_semantics: dict[str, Any] = {
            "is_commutative": op in {"Add", "Mul", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Equal", "NotEqual"},
            "is_associative": op in {"Add", "Mul", "BitwiseAnd", "BitwiseOr", "BitwiseXor"},
            "is_idempotent": op in {"Relu", "Floor", "Ceil", "Round", "Abs"},
            "differentiable": domain not in {"control_flow", "creation"} and op not in {"Equal", "NotEqual", "Greater", "Less", "GreaterEqual", "LessEqual", "ArgMax", "ArgMin"},
        }
        if op == "Add":
            math_semantics["identity_element"] = 0
        elif op == "Mul":
            math_semantics["identity_element"] = 1

        # Signature
        args_list = []
        for inp_name in default_inputs:
            args_list.append(
                {
                    "name": inp_name,
                    "type": "Tensor",
                    "is_variadic": False,
                }
            )

        shape_formula = "broadcast(x1.shape, x2.shape)" if is_binary and len(default_inputs) >= 2 else (f"{default_inputs[0]}.shape" if default_inputs else "shape")
        dtype_formula = "promote(x1.dtype, x2.dtype)" if is_binary and len(default_inputs) >= 2 else (f"{default_inputs[0]}.dtype" if default_inputs else "float32")

        signature: dict[str, Any] = {
            "args": args_list,
            "output_shape_formula": shape_formula,
            "output_dtype_formula": dtype_formula,
        }

        op_def = {
            "operation": op,
            "description": base_data.get("description", f"Declarative {op} operation in {domain} domain."),
            "domain": domain,
            "input_tensors": default_inputs,
            "output_tensors": ["out"],
            "broadcast_semantics": "numpy",
            "type_promotion": "standard",
            "math_semantics": math_semantics,
            "signature": signature,
            "variants": variants,
        }
        if "autodiff" in base_data:
            op_def["autodiff"] = base_data["autodiff"]

        # Validate against OperationDefinitionModel
        OperationDefinitionModel.model_validate(op_def)
        domain_dict[op] = op_def

    # Validate against DomainOperationsModel
    DomainOperationsModel.model_validate(domain_dict)

    out_path = os.path.join(BASE_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(domain_dict, f, sort_keys=True, indent=2)

    return domain_dict


def main() -> None:
    """Migrate all 13 domain YAML definition files."""
    for filename, spec in DOMAIN_SPECS.items():
        domain_dict = migrate_domain(filename, spec)
        print(f"Generated {filename} with {len(domain_dict)} operations.")


if __name__ == "__main__":
    main()
