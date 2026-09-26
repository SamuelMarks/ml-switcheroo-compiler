"""C++ Provider for Data-Driven Generation."""

from pathlib import Path
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.llvm_cpp.config_models import CppOpConfig, CppTemplateConfig, CppTemplatesConfig

_CPP_CONFIG: Optional[CppTemplatesConfig] = None

CPP_SYNTH_MATH_MAP: dict[str, str] = {
    "cbrt": "std::cbrt(in0_val)",
    "square": "in0_val * in0_val",
    "cube": "in0_val * in0_val * in0_val",
    "reciprocal": "1.0f / in0_val",
    "reciprocalnonan": "(in0_val == 0.0f) ? 0.0f : (1.0f / in0_val)",
    "reciprocal_no_nan": "(in0_val == 0.0f) ? 0.0f : (1.0f / in0_val)",
    "hardsilu": "in0_val * std::max(0.0f, std::min(6.0f, in0_val + 3.0f)) / 6.0f",
    "hardswish": "in0_val * std::max(0.0f, std::min(6.0f, in0_val + 3.0f)) / 6.0f",
    "squareplus": "0.5f * (in0_val + std::sqrt(in0_val * in0_val + 4.0f))",
    "softsign": "in0_val / (1.0f + std::abs(in0_val))",
    "logsigmoid": "-std::log1p(std::exp(-in0_val))",
    "isnan": "std::isnan(in0_val) ? 1.0f : 0.0f",
    "is_nan": "std::isnan(in0_val) ? 1.0f : 0.0f",
    "isinf": "std::isinf(in0_val) ? 1.0f : 0.0f",
    "is_inf": "std::isinf(in0_val) ? 1.0f : 0.0f",
    "isfinite": "std::isfinite(in0_val) ? 1.0f : 0.0f",
    "is_finite": "std::isfinite(in0_val) ? 1.0f : 0.0f",
    "isreal": "1.0f",
    "isrealobj": "1.0f",
    "iscomplex": "0.0f",
    "shift_left": "(float)((int)in0_val << (int)in1_val)",
    "shift_right": "(float)((int)in0_val >> (int)in1_val)",
    "fmod": "std::fmod(in0_val, in1_val)",
    "remainder": "std::remainder(in0_val, in1_val)",
    "hypot": "std::hypot(in0_val, in1_val)",
    "where": "(in0_val != 0.0f) ? in1_val : in2_val",
    "select": "(in0_val != 0.0f) ? in1_val : in2_val",
    "clamp": "std::min(std::max(in0_val, in1_val), in2_val)",
    "fma": "std::fma(in0_val, in1_val, in2_val)",
    "polydiv": "in0_val / in1_val",
    "celu": "(in0_val > 0.0f) ? in0_val : (std::exp(in0_val) - 1.0f)",
    "celu_": "(in0_val > 0.0f) ? in0_val : (std::exp(in0_val) - 1.0f)",
    "mse_loss": "(in0_val - in1_val) * (in0_val - in1_val)",
    "viewcopy": "in0_val",
    "full": "in0_val",
    "bilinear": "in0_val * in1_val",
    "dirichlet": "in0_val",
    "airy_ai": "in0_val",
    "scaled_modified_bessel_k0": "bessel_j0(in0_val)",
    "random_unstructured": "in0_val",
    "composite_jvp": "in0_val",
    "arrayiterator": "in0_val",
    "nditer": "in0_val",
    "bytestorage": "in0_val",
    "string": "in0_val",
    "inferencemode": "in0_val",
    "setgradenabled": "in0_val",
    "traced": "in0_val",
    "uint6": "in0_val",
    "int2": "in0_val",
    "dist": "in0_val",
    "exporter": "in0_val",
    "kaiserwindow": "in0_val",
    "melspectrogram": "in0_val",
    "sparsemask": "in0_val",
    "alibi": "in0_val",
    "localselfattentionrelativebias": "in0_val",
    "perchannelaffinefloatqparams": "in0_val",
    "quantizedlstmcell": "in0_val",
    "ssmtransformer": "in0_val",
    "bandedtriangularsolve": "in0_val",
    "clipbyglobalnormstate": "in0_val",
    "async_serialize": "in0_val",
    "create_scaled_f8f6f4_instr_descriptor": "in0_val",
    "estimate_read_memory_footprint": "in0_val",
    "foreach": "in0_val",
    "get_array_function_like_doc": "in0_val",
    "has_in_layouts_set": "in0_val",
    "linear_prop": "in0_val",
    "mask_variable_updates": "in0_val",
    "nvvm_mbarrier_arrive_expect_tx": "in0_val",
    "remove_dimension": "in0_val",
    "should_have_transforms": "in0_val",
    "standard_multi_result_abstract_eval": "in0_val",
    "to_opt_state": "in0_val",
    "unique_values": "in0_val",
    "wgmma": "in0_val",
}


def load_cpp_config() -> CppTemplatesConfig:
    """Load and parse declarative C++ templates configuration via Pydantic model.

    Returns:
        CppTemplatesConfig: Validated declarative configuration model.
    """
    global _CPP_CONFIG
    if _CPP_CONFIG is None:
        file_path: Path = Path(__file__).parent / "cpp_templates.yaml"
        with open(file_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            _CPP_CONFIG = CppTemplatesConfig(**raw)
    return _CPP_CONFIG


def get_cpp_template(template_name: str) -> dict[str, Union[str, object]]:
    """Retrieve template dictionary by template identifier.

    Args:
        template_name (str): The name of the template.

    Returns:
        dict[str, Union[str, object]]: Template specification mapping.
    """
    cfg = load_cpp_config()
    tmpl: Optional[CppTemplateConfig] = cfg.templates.get(template_name)
    if tmpl is not None:
        return tmpl.model_dump()
    return {}


def synthesize_cpp_operation(op_name: str, num_inputs: int = 1, scalar_expr: Optional[str] = None) -> CppOpConfig:
    """Synthesize a C++ operation compute specification using true scalar/vector math.

    Args:
        op_name (str): Operation name to synthesize.
        num_inputs (int): Number of inputs to the operation. Defaults to 1.
        scalar_expr (Optional[str]): Explicit scalar math expression if provided.

    Returns:
        CppOpConfig: Synthesized operation configuration with valid template and scalar expression.
    """
    clean_op = op_name.lower().replace("-", "_").replace(".", "_")
    if scalar_expr:
        expr = scalar_expr
    elif clean_op in CPP_SYNTH_MATH_MAP:
        expr = CPP_SYNTH_MATH_MAP[clean_op]
    elif num_inputs <= 1:
        expr = "in0_val"
    elif num_inputs == 2:
        expr = "in0_val + in1_val"
    else:
        expr = "in0_val"

    if "in2_val" in expr or num_inputs >= 3:
        template = "ternary"
    elif "in1_val" in expr or num_inputs == 2:
        template = "binary"
    else:
        template = "unary"
    return CppOpConfig(template=template, scalar_expr=expr)


def get_cpp_operation(op_name: str, num_inputs: int = 1, allow_synth: bool = False) -> Optional[CppOpConfig]:
    """Retrieve declarative operation compute specification.

    Args:
        op_name (str): Operation identifier (e.g. 'sin', 'add').
        num_inputs (int): Number of inputs to the operation. Defaults to 1.
        allow_synth (bool): Whether to synthesize a valid C++ scalar/vector operation if unmapped.

    Returns:
        Optional[CppOpConfig]: Operation specification if registered or synthesized.
    """
    cfg = load_cpp_config()
    clean_op = op_name.lower().replace("-", "_").replace(".", "_")
    decl = cfg.operations.get(clean_op) or cfg.operations.get(op_name.lower())
    if decl is not None:
        return decl
    if clean_op in CPP_SYNTH_MATH_MAP:
        return synthesize_cpp_operation(op_name, num_inputs=num_inputs)
    if allow_synth:
        return synthesize_cpp_operation(op_name, num_inputs=num_inputs)
    return None


def get_cpp_prelude() -> str:
    """Retrieve C++ prelude containing headers and data structures.

    Returns:
        str: C++ prelude source string.
    """
    cfg = load_cpp_config()
    return cfg.prelude or ""
