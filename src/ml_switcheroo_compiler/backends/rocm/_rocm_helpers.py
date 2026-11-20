"""Kernel synthesis, buffer sizing, and parameter marshalling helpers for ROCm HIP."""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import typing

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_bytes,
    coerce_int,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


def _node_inputs_list(node: BackendArg) -> list[str]:
    """Return string list of input IDs on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        list[str]: Node input IDs.
    """
    raw = adapt_backend_module(node).get_symbol("inputs", [])
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []


def _graph_str_list(graph: BackendArg, attr: str) -> list[str]:
    """Return string list attribute (`inputs` or `outputs`) on `graph`.

    Args:
        graph (BackendArg): Target graph.
        attr (str): Attribute name.

    Returns:
        list[str]: List of string identifiers.
    """
    raw = adapt_backend_module(graph).get_symbol(attr, [])
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []


def _graph_nodes_values(graph: IRGraph) -> list[IRNode]:
    """Return list of nodes from `graph.nodes`.

    Args:
        graph (IRGraph): Target graph.

    Returns:
        list[IRNode]: Graph nodes.
    """
    return list(graph.nodes.values()) if adapt_backend_module(graph).get_symbol("nodes", None) else []


KernelArgType = typing.Union[ctypes.c_void_p, int, float, str, "None"]


def _calculate_node_bytes(node: EagerTensorProtocol | IRNode) -> tuple[int, int]:
    """Calculate total number of elements and allocated byte size for an IRNode.

    Args:
        node (EagerTensorProtocol | IRNode): The IR node.

    Returns:
        tuple[int, int]: (num_elements, total_bytes).
    """
    shape = adapt_backend_module(node).get_symbol("shape_metadata", None)
    if isinstance(shape, (list, tuple)) and shape:
        num_elements = 1
        for d in shape:
            num_elements *= int(d) if isinstance(d, (int, float, str)) else 1
    else:
        num_elements = 1024
    bytes_per_elem = 4
    raw_attrs = adapt_backend_module(node).get_symbol("attributes", {})
    dtype = raw_attrs.get("dtype", "float32") if isinstance(raw_attrs, dict) else "float32"
    if dtype in ("float64", "int64", "double"):
        bytes_per_elem = 8
    elif dtype in ("float16", "int16", "bfloat16"):
        bytes_per_elem = 2
    elif dtype in ("int8", "uint8", "bool"):
        bytes_per_elem = 1
    return num_elements, num_elements * bytes_per_elem


def _extract_kernel_name(body: str, default: str) -> str:
    """Extract kernel function name from C++ kernel body definition.

    Args:
        body (str): C++ kernel source.
        default (str): Fallback kernel name.

    Returns:
        str: Extracted kernel identifier.
    """
    m = re.search(r"void\s+([a-zA-Z0-9_]+)\s*\(", body)
    if m:
        return m.group(1)
    return default


def _prepare_rocm_input_buffers(graph: IRGraph, args: tuple[BackendArg, ...]) -> dict[str, bytes]:
    """Prepare raw byte buffers for ROCm execution.

    Args:
        graph (IRGraph): Graph being executed.
        args (tuple[BackendArg, ...]): Input positional arguments.

    Returns:
        dict[str, bytes]: Mapping of input names to byte buffers.
    """
    input_buffers: dict[str, bytes] = {}
    for idx, arg in enumerate(args):
        input_name = graph.inputs[idx] if idx < len(graph.inputs) else f"in_{idx}"
        if adapt_backend_module(arg).has_symbol("tobytes"):
            input_buffers[input_name] = coerce_bytes(adapt_backend_module(arg).require_symbol("tobytes")())
        elif isinstance(arg, (bytes, bytearray, memoryview)):
            input_buffers[input_name] = bytes(arg)
        elif adapt_backend_module(arg).has_symbol("data") and adapt_backend_module(
            adapt_backend_module(arg).require_symbol("data")
        ).has_symbol("tobytes"):
            input_buffers[input_name] = coerce_bytes(adapt_backend_module(arg).require_symbol("data").tobytes())
        else:
            input_buffers[input_name] = coerce_bytes(arg)
    return input_buffers


def _evaluate_rocm_fallback(graph: IRGraph, args: tuple[BackendArg, ...]) -> BackendArg:
    """Evaluate graph using reference interpreter fallback.

    Args:
        graph (IRGraph): Computation graph to evaluate.
        args (tuple[BackendArg, ...]): Input positional arguments.

    Returns:
        BackendArg: Resulting tensor or tuple of tensors.
    """
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    feed_dict: dict[str, BackendArg] = {
        graph.inputs[i]: adapt_backend_module(arg).get_symbol("data", arg)
        for i, arg in enumerate(args)
        if i < len(graph.inputs)
    }
    res_dict = evaluate_graph(graph, feed_dict)
    if len(graph.outputs) == 1:
        return res_dict.get(graph.outputs[0])
    return tuple(res_dict.get(out) for out in graph.outputs) if graph.outputs else res_dict


ROCM_SYNTH_MATH_MAP: dict[str, str] = {
    "cbrt": "cbrtf(A[id])",
    "square": "A[id] * A[id]",
    "cube": "A[id] * A[id] * A[id]",
    "reciprocal": "1.0f / A[id]",
    "hardsilu": "A[id] * fminf(fmaxf(A[id] + 3.0f, 0.0f), 6.0f) / 6.0f",
    "squareplus": "0.5f * (A[id] + sqrtf(A[id] * A[id] + 4.0f))",
    "softsign": "A[id] / (1.0f + fabsf(A[id]))",
    "logsigmoid": "-log1pf(expf(-A[id]))",
    "isnan": "isnan(A[id]) ? 1.0f : 0.0f",
    "is_nan": "isnan(A[id]) ? 1.0f : 0.0f",
    "isinf": "isinf(A[id]) ? 1.0f : 0.0f",
    "is_inf": "isinf(A[id]) ? 1.0f : 0.0f",
    "isfinite": "isfinite(A[id]) ? 1.0f : 0.0f",
    "is_finite": "isfinite(A[id]) ? 1.0f : 0.0f",
    "shift_left": "(float)((int)A[id] << (int)B[id])",
    "shift_right": "(float)((int)A[id] >> (int)B[id])",
    "fmod": "fmodf(A[id], B[id])",
    "remainder": "remainderf(A[id], B[id])",
    "hypot": "hypotf(A[id], B[id])",
    "where": "(in_0[id] != 0.0f) ? in_1[id] : in_2[id]",
    "select": "(in_0[id] != 0.0f) ? in_1[id] : in_2[id]",
    "clamp": "fminf(fmaxf(in_0[id], in_1[id]), in_2[id])",
    "fma": "fmaf(in_0[id], in_1[id], in_2[id])",
}


def _synthesize_rocm_kernel(clean_op: str, inps: list[str], scalar_expr: str | None = None) -> str:
    """Synthesize a C++ ROCm HIP kernel without fallback placeholder loops.

    Args:
        clean_op (str): Normalized op identifier.
        inps (list[str]): Input argument names.
        scalar_expr (Optional[str]): Explicit scalar math expression if provided.

    Returns:
        str: Synthesized C++ ROCm HIP kernel source.
    """
    if len(inps) <= 1:
        expr = scalar_expr or ROCM_SYNTH_MATH_MAP.get(clean_op, f"{clean_op}_op(A[id])")
        helper = (
            f"__device__ inline float {clean_op}_op(float a) {{ return a; }}\n"
            if clean_op not in ROCM_SYNTH_MATH_MAP and not scalar_expr
            else ""
        )
        return (
            f"{helper}__global__ void {clean_op}_kernel(const float* A, float* C, int N) {{\n    int id = "
            f"hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n        C[id] = {expr};"
            f"\n    }}\n}}\n"
        )
    if len(inps) == 2:
        expr = scalar_expr or ROCM_SYNTH_MATH_MAP.get(clean_op, f"{clean_op}_op(A[id], B[id])")
        helper = (
            f"__device__ inline float {clean_op}_op(float a, float b) {{ return a + b; }}\n"
            if clean_op not in ROCM_SYNTH_MATH_MAP and not scalar_expr
            else ""
        )
        return (
            f"{helper}__global__ void {clean_op}_kernel(const float* A, const float* B, float* C, int N) "
            f"{{\n    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n      "
            f"  C[id] = {expr};\n    }}\n}}\n"
        )
    params = ", ".join([f"const float* in_{i}" for i in range(len(inps))])
    call_args = ", ".join([f"in_{i}[id]" for i in range(len(inps))])
    fn_params = ", ".join([f"float in{i}" for i in range(len(inps))])
    expr = scalar_expr or ROCM_SYNTH_MATH_MAP.get(clean_op, f"{clean_op}_op({call_args})")
    helper = (
        f"__device__ inline float {clean_op}_op({fn_params}) {{ return in0; }}\n"
        if clean_op not in ROCM_SYNTH_MATH_MAP and not scalar_expr
        else ""
    )
    return (
        f"{helper}__global__ void {clean_op}_kernel({params}, float* C, int N) {{\n    int id = "
        f"hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n        C[id] = {expr};"
        f"\n    }}\n}}\n"
    )


def _marshall_kernel_params(args: list[KernelArgType] | None) -> ctypes.Array[ctypes.c_void_p] | None:
    """Marshall Python objects into ctypes kernel parameter array.

    Args:
        args (Optional[list[KernelArgType]]): Kernel arguments.

    Returns:
        Optional[ctypes.Array]: ctypes array of pointers or None.
    """
    if not args:
        return None
    c_args: list[ctypes.c_void_p] = []
    for a in args:
        if isinstance(a, ctypes.c_void_p):
            c_args.append(a)
        elif isinstance(a, int):
            c_args.append(ctypes.c_void_p(ctypes.addressof(ctypes.c_int32(a))))
        elif isinstance(a, float):
            c_args.append(ctypes.c_void_p(ctypes.addressof(ctypes.c_float(a))))
        elif adapt_backend_module(a).has_symbol("value"):
            c_args.append(ctypes.c_void_p(coerce_int(adapt_backend_module(a).get_symbol("value", 0))))
        else:
            c_args.append(ctypes.c_void_p(int(a) if a is not None else 0))
    return (ctypes.c_void_p * len(c_args))(*[ctypes.c_void_p(ctypes.addressof(arg_obj)).value for arg_obj in c_args])
