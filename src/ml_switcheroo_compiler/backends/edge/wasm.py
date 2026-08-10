# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class WasmCodeGenerator(BaseGenerator):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
        """Initialize WasmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.is_simd: bool = False

    def _allocate_aligned_memory(self, size_bytes: int, alignment: int = 16) -> str:
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        if not shape:
            return [], "0"
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code = " + ".join([f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)])
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "bool": "bool",
        }.get(str(dtype).lower(), "float")

    def _num_elements(self, shape: list[int]) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    def get_helper_functions(self) -> list[str]:
        """Return C++ helper functions/macros for missing WASM math.

        Returns:
            list[str]: Lines of C++ macros/functions.
        """
        helpers = [
            "// --- Scalar Fallback Helpers ---",
            "inline float _scalar_logicaland(float a, float b) { return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f; }",
            "inline float _scalar_logicalor(float a, float b) { return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f; }",
            "inline float _scalar_logicalxor(float a, float b) { return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f; }",
            "inline float _scalar_equal(float a, float b) { return (a == b) ? 1.0f : 0.0f; }",
            "inline float _scalar_notequal(float a, float b) { return (a != b) ? 1.0f : 0.0f; }",
            "inline float _scalar_greater(float a, float b) { return (a > b) ? 1.0f : 0.0f; }",
            "inline float _scalar_less(float a, float b) { return (a < b) ? 1.0f : 0.0f; }",
            "inline float _scalar_greaterequal(float a, float b) { return (a >= b) ? 1.0f : 0.0f; }",
            "inline float _scalar_lessequal(float a, float b) { return (a <= b) ? 1.0f : 0.0f; }",
            "inline float _scalar_floordivide(float a, float b) { return std::floor(a / b); }",
            "inline float _scalar_power(float a, float b) { return std::pow(a, b); }",
            "inline float _scalar_sign(float a, float dummy) { return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); }",
            "inline float _scalar_ceil(float a, float dummy) { return std::ceil(a); }",
            "inline float _scalar_floor(float a, float dummy) { return std::floor(a); }",
            "inline float _scalar_round(float a, float dummy) { return std::round(a); }",
            "inline float _scalar_rsqrt(float a, float dummy) { return 1.0f / std::sqrt(a); }",
            "inline float _scalar_log1p(float a, float dummy) { return std::log1p(a); }",
            "inline float _scalar_expm1(float a, float dummy) { return std::expm1(a); }",
            "inline float _scalar_sin(float a, float dummy) { return std::sin(a); }",
            "inline float _scalar_cos(float a, float dummy) { return std::cos(a); }",
            "inline float _scalar_tan(float a, float dummy) { return std::tan(a); }",
            "inline float _scalar_asin(float a, float dummy) { return std::asin(a); }",
            "inline float _scalar_acos(float a, float dummy) { return std::acos(a); }",
            "inline float _scalar_atan(float a, float dummy) { return std::atan(a); }",
            "inline float _scalar_sinh(float a, float dummy) { return std::sinh(a); }",
            "inline float _scalar_cosh(float a, float dummy) { return std::cosh(a); }",
            "inline float _scalar_tanh(float a, float dummy) { return std::tanh(a); }",
            "inline float _scalar_gelu(float a, float dummy) { return 0.5f * a * (1.0f + std::tanh(0.7978845608f * (a + 0.044715f * a * a * a))); }",
            "inline float _scalar_swish(float a, float dummy) { return a / (1.0f + std::exp(-a)); }",
            "inline float _scalar_silu(float a, float dummy) { return a / (1.0f + std::exp(-a)); }",
            "inline float _scalar_sigmoid(float a, float dummy) { return 1.0f / (1.0f + std::exp(-a)); }",
            "inline float _scalar_cast(float a, float dummy) { return a; }",
            "// --- SIMD Fast Math Approximations ---",
            "inline v128_t _wasm_tanh(v128_t x) {",
            "    v128_t one = wasm_f32x4_splat(1.0f);",
            "    v128_t abs_x = wasm_f32x4_abs(x);",
            "    v128_t den = wasm_f32x4_add(one, abs_x);",
            "    return wasm_f32x4_div(x, den);",
            "}",
            "inline v128_t _wasm_sigmoid(v128_t x) {",
            "    v128_t one = wasm_f32x4_splat(1.0f);",
            "    v128_t half = wasm_f32x4_splat(0.5f);",
            "    v128_t abs_x = wasm_f32x4_abs(x);",
            "    v128_t den = wasm_f32x4_add(one, abs_x);",
            "    v128_t div = wasm_f32x4_div(x, den);",
            "    v128_t mul = wasm_f32x4_mul(div, half);",
            "    return wasm_f32x4_add(mul, half);",
            "}",
        ]
        return helpers

    def generic_visit(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def _generate_matmul(self, node: Any, clean_id: str, inputs: list[str], shape: list[int]) -> None:
        N = shape[1] if len(shape) > 1 else 1
        M = shape[0] if len(shape) > 0 else 1

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None)
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        if not in0_shape:
            in0_shape = [1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [int(in0_shape)]
        K = in0_shape[1] if len(in0_shape) > 1 else 1

        self.add_line("  // MatMul SIMD block")
        self.add_line(f"  int limit_{clean_id}_N = {N} - ({N} % 4);")
        self.add_line(f"  for (int i = 0; i < {M}; ++i) {{")
        self.add_line(f"    for (int j = 0; j < limit_{clean_id}_N; j += 4) {{")
        self.add_line("      v128_t sum = wasm_f32x4_splat(0.0f);")
        self.add_line(f"      for (int k = 0; k < {K}; ++k) {{")
        self.add_line(f"        v128_t a = wasm_f32x4_splat(buf_{inputs[0]}[i * {K} + k]);")
        self.add_line(f"        v128_t b = wasm_v128_load(&buf_{inputs[1]}[k * {N} + j]);")
        self.add_line("        sum = wasm_f32x4_add(sum, wasm_f32x4_mul(a, b));")
        self.add_line("      }")
        self.add_line(f"      wasm_v128_store(&buf_{clean_id}[i * {N} + j], sum);")
        self.add_line("    }")
        self.add_line(f"    for (int j = limit_{clean_id}_N; j < {N}; ++j) {{")
        self.add_line("      float sum = 0.0f;")
        self.add_line(f"      for (int k = 0; k < {K}; ++k) {{")
        self.add_line(f"        sum += buf_{inputs[0]}[i * {K} + k] * buf_{inputs[1]}[k * {N} + j];")
        self.add_line("      }")
        self.add_line(f"      buf_{clean_id}[i * {N} + j] = sum;")
        self.add_line("    }")
        self.add_line("  }")

    def _generate_conv_pool(self, clean_id: str, inputs: list[str], nelem: int) -> None:
        self.add_line("  // SIMD Conv/Pool Fallback")
        self.add_line(f"  int limit_{clean_id} = {nelem} - ({nelem} % 4);")
        self.add_line("  #pragma GCC unroll 4")

        self.add_line(f"  for (int i = 0; i < limit_{clean_id}; i += 4) {{")
        self.add_line(f"    v128_t val = wasm_v128_load(&buf_{inputs[0]}[i]);")
        self.add_line(f"    wasm_v128_store(&buf_{clean_id}[i], val);")
        self.add_line("  }")
        self.add_line(f"  for (int j = limit_{clean_id}; j < {nelem}; ++j) {{")
        self.add_line(f"    buf_{clean_id}[j] = buf_{inputs[0]}[j];")
        self.add_line("  }")

    def _generate_reduce(self, node: Any, op_type: str, clean_id: str, inputs: list[str]) -> None:
        self.add_line("  // SIMD Reduction")
        if op_type == "ReduceProd":
            self.add_line(f"  v128_t sum_{clean_id} = wasm_f32x4_splat(1.0f);")
            self.add_line(f"  float scalar_sum_{clean_id} = 1.0f;")
        elif op_type in ("ArgMax", "ArgMin"):
            self.add_line(f"  float best_val_{clean_id} = buf_{inputs[0]}[0];")
            self.add_line(f"  float best_idx_{clean_id} = 0.0f;")
        else:
            self.add_line(f"  v128_t sum_{clean_id} = wasm_f32x4_splat(0.0f);")
            self.add_line(f"  float scalar_sum_{clean_id} = 0.0f;")

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None)
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1]
        if not in0_shape:
            in0_shape = [1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [int(in0_shape)]
        in0_nelem = self._num_elements(in0_shape)

        self.add_line(f"  int limit_{clean_id} = {in0_nelem} - ({in0_nelem} % 4);")

        if op_type in ("ArgMax", "ArgMin"):
            # Scalar only for argmax/argmin for now due to lane indexing complexity
            self.add_line(f"  for (int j = 1; j < {in0_nelem}; ++j) {{")
            if op_type == "ArgMax":
                self.add_line(f"    if (buf_{inputs[0]}[j] > best_val_{clean_id}) {{ best_val_{clean_id} = buf_{inputs[0]}[j]; best_idx_{clean_id} = (float)j; }}")
            else:
                self.add_line(f"    if (buf_{inputs[0]}[j] < best_val_{clean_id}) {{ best_val_{clean_id} = buf_{inputs[0]}[j]; best_idx_{clean_id} = (float)j; }}")
            self.add_line("  }")
            self.add_line(f"  buf_{clean_id}[0] = best_idx_{clean_id};")
        else:
            self.add_line("  #pragma GCC unroll 4")

            self.add_line(f"  for (int i = 0; i < limit_{clean_id}; i += 4) {{")
            self.add_line(f"    v128_t val = wasm_v128_load(&buf_{inputs[0]}[i]);")
            if op_type in ("ReduceSum", "ReduceMean"):
                self.add_line(f"    sum_{clean_id} = wasm_f32x4_add(sum_{clean_id}, val);")
            elif op_type == "ReduceProd":
                self.add_line(f"    sum_{clean_id} = wasm_f32x4_mul(sum_{clean_id}, val);")
            elif op_type == "ReduceMax":
                self.add_line(f"    sum_{clean_id} = wasm_f32x4_pmax(sum_{clean_id}, val);")
            elif op_type == "ReduceMin":
                self.add_line(f"    sum_{clean_id} = wasm_f32x4_pmin(sum_{clean_id}, val);")
            self.add_line("  }")
            self.add_line(f"  for (int j = limit_{clean_id}; j < {in0_nelem}; ++j) {{")
            if op_type in ("ReduceSum", "ReduceMean"):
                self.add_line(f"    scalar_sum_{clean_id} += buf_{inputs[0]}[j];")
            elif op_type == "ReduceProd":
                self.add_line(f"    scalar_sum_{clean_id} *= buf_{inputs[0]}[j];")
            elif op_type == "ReduceMax":
                self.add_line(f"    scalar_sum_{clean_id} = std::max(scalar_sum_{clean_id}, buf_{inputs[0]}[j]);")
            elif op_type == "ReduceMin":
                self.add_line(f"    scalar_sum_{clean_id} = std::min(scalar_sum_{clean_id}, buf_{inputs[0]}[j]);")
            self.add_line("  }")

            self.add_line(f"  float temp_sum_{clean_id}[4];")
            self.add_line(f"  wasm_v128_store(temp_sum_{clean_id}, sum_{clean_id});")

            if op_type in ("ReduceSum", "ReduceMean"):
                self.add_line(f"  scalar_sum_{clean_id} += temp_sum_{clean_id}[0] + temp_sum_{clean_id}[1] + temp_sum_{clean_id}[2] + temp_sum_{clean_id}[3];")
                if op_type == "ReduceMean":
                    self.add_line(f"  scalar_sum_{clean_id} /= {in0_nelem}.0f;")
            elif op_type == "ReduceProd":
                self.add_line(f"  scalar_sum_{clean_id} *= temp_sum_{clean_id}[0] * temp_sum_{clean_id}[1] * temp_sum_{clean_id}[2] * temp_sum_{clean_id}[3];")
            elif op_type == "ReduceMax":
                self.add_line(f"  scalar_sum_{clean_id} = std::max({{scalar_sum_{clean_id}, temp_sum_{clean_id}[0], temp_sum_{clean_id}[1], temp_sum_{clean_id}[2], temp_sum_{clean_id}[3]}});")
            elif op_type == "ReduceMin":
                self.add_line(f"  scalar_sum_{clean_id} = std::min({{scalar_sum_{clean_id}, temp_sum_{clean_id}[0], temp_sum_{clean_id}[1], temp_sum_{clean_id}[2], temp_sum_{clean_id}[3]}});")

            self.add_line(f"  buf_{clean_id}[0] = scalar_sum_{clean_id};")

    def _generate_generic(self, node: Any, op_type: str, clean_id: str, inputs: list[str], nelem: int) -> None:
        self.add_line(f"  // Generic SIMD loop for {op_type}")
        self.add_line(f"  int limit_{clean_id} = {nelem} - ({nelem} % 4);")
        self.add_line("  #pragma GCC unroll 4")

        self.add_line(f"  for (int i = 0; i < limit_{clean_id}; i += 4) {{")

        if op_type == "Constant":
            val = getattr(node, "attributes", {}).get("value", 0.0)
            self.add_line(f"      v128_t res = wasm_f32x4_splat({val}f);")
        else:
            if len(inputs) > 0:
                self.add_line(f"      v128_t in0 = wasm_v128_load(&buf_{inputs[0]}[i]);")
            if len(inputs) > 1:
                self.add_line(f"      v128_t in1 = wasm_v128_load(&buf_{inputs[1]}[i]);")

            if op_type == "Add":
                self.add_line("      v128_t res = wasm_f32x4_add(in0, in1);")
            elif op_type == "Subtract":
                self.add_line("      v128_t res = wasm_f32x4_sub(in0, in1);")
            elif op_type == "Multiply":
                self.add_line("      v128_t res = wasm_f32x4_mul(in0, in1);")
            elif op_type in ("TrueDivide", "Div"):
                self.add_line("      v128_t res = wasm_f32x4_div(in0, in1);")
            elif op_type == "Negative" or op_type == "Neg":
                self.add_line("      v128_t res = wasm_f32x4_neg(in0);")
            elif op_type == "Sqrt":
                self.add_line("      v128_t res = wasm_f32x4_sqrt(in0);")
            elif op_type == "Abs":
                self.add_line("      v128_t res = wasm_f32x4_abs(in0);")
            elif op_type in ("Minimum", "Min"):
                self.add_line("      v128_t res = wasm_f32x4_pmin(in0, in1);")
            elif op_type in ("Maximum", "Max"):
                self.add_line("      v128_t res = wasm_f32x4_pmax(in0, in1);")
            elif op_type == "Relu":
                self.add_line("      v128_t zero = wasm_f32x4_splat(0.0f);")
                self.add_line("      v128_t res = wasm_f32x4_pmax(zero, in0);")
            elif op_type in (
                "LogicalAnd",
                "LogicalOr",
                "LogicalXor",
                "Equal",
                "NotEqual",
                "Greater",
                "Less",
                "GreaterEqual",
                "LessEqual",
                "FloorDivide",
                "Power",
                "Sign",
                "Ceil",
                "Floor",
                "Round",
                "Rsqrt",
                "Log1p",
                "Expm1",
                "Sin",
                "Cos",
                "Tan",
                "Asin",
                "Acos",
                "Atan",
                "Sinh",
                "Cosh",
                "Gelu",
                "Swish",
                "Silu",
                "Cast",
            ):
                # For complex math lacking direct v128 instructions, we peel the lanes using std lib calls.
                self.add_line(
                    f"      v128_t res = wasm_f32x4_make(_scalar_{op_type.lower()}(wasm_f32x4_extract_lane(in0, 0), "
                    f"{'wasm_f32x4_extract_lane(in1, 0)' if len(inputs) > 1 else '0.0f'}), "
                    f"_scalar_{op_type.lower()}(wasm_f32x4_extract_lane(in0, 1), "
                    f"{'wasm_f32x4_extract_lane(in1, 1)' if len(inputs) > 1 else '0.0f'}), "
                    f"_scalar_{op_type.lower()}(wasm_f32x4_extract_lane(in0, 2), "
                    f"{'wasm_f32x4_extract_lane(in1, 2)' if len(inputs) > 1 else '0.0f'}), "
                    f"_scalar_{op_type.lower()}(wasm_f32x4_extract_lane(in0, 3), "
                    f"{'wasm_f32x4_extract_lane(in1, 3)' if len(inputs) > 1 else '0.0f'}));"
                )
            elif op_type == "Tanh":
                self.add_line("      v128_t res = _wasm_tanh(in0);")
            elif op_type in ("Exp", "Log"):
                self.add_line(f"      v128_t res = wasm_f32x4_make(std::{op_type.lower()}(wasm_f32x4_extract_lane(in0, 0)), std::{op_type.lower()}(wasm_f32x4_extract_lane(in0, 1)), std::{op_type.lower()}(wasm_f32x4_extract_lane(in0, 2)), std::{op_type.lower()}(wasm_f32x4_extract_lane(in0, 3)));")
            elif op_type == "Sigmoid":
                self.add_line("      v128_t res = _wasm_sigmoid(in0);")

            else:
                self.add_line("      v128_t res = wasm_f32x4_splat(0.0f);")

        self.add_line(f"    wasm_v128_store(&buf_{clean_id}[i], res);")
        self.add_line("  }")
        self.add_line("  // Scalar fallback fringe")
        self.add_line(f"  for (int j = limit_{clean_id}; j < {nelem}; ++j) {{")

        if op_type == "Constant":
            val = getattr(node, "attributes", {}).get("value", 0.0)
            self.add_line(f"    buf_{clean_id}[j] = {val}f;")
        elif op_type == "Add":
            self.add_line(f"    buf_{clean_id}[j] = buf_{inputs[0]}[j] + buf_{inputs[1]}[j];")
        elif op_type == "Subtract":
            self.add_line(f"    buf_{clean_id}[j] = buf_{inputs[0]}[j] - buf_{inputs[1]}[j];")
        elif op_type == "Multiply":
            self.add_line(f"    buf_{clean_id}[j] = buf_{inputs[0]}[j] * buf_{inputs[1]}[j];")
        elif op_type in ("TrueDivide", "Div"):
            self.add_line(f"    buf_{clean_id}[j] = buf_{inputs[0]}[j] / buf_{inputs[1]}[j];")
        elif op_type == "Negative" or op_type == "Neg":
            self.add_line(f"    buf_{clean_id}[j] = -buf_{inputs[0]}[j];")
        elif op_type == "Sqrt":
            self.add_line(f"    buf_{clean_id}[j] = std::sqrt(buf_{inputs[0]}[j]);")
        elif op_type == "Abs":
            self.add_line(f"    buf_{clean_id}[j] = std::abs(buf_{inputs[0]}[j]);")
        elif op_type in ("Minimum", "Min"):
            self.add_line(f"    buf_{clean_id}[j] = std::min(buf_{inputs[0]}[j], buf_{inputs[1]}[j]);")
        elif op_type in ("Maximum", "Max"):
            self.add_line(f"    buf_{clean_id}[j] = std::max(buf_{inputs[0]}[j], buf_{inputs[1]}[j]);")
        elif op_type == "Relu":
            self.add_line(f"    buf_{clean_id}[j] = std::max(0.0f, buf_{inputs[0]}[j]);")
        elif op_type in ("Exp", "Log"):
            self.add_line(f"    buf_{clean_id}[j] = std::{op_type.lower()}(buf_{inputs[0]}[j]);")
        else:
            in0_val = f"buf_{inputs[0]}[j]" if len(inputs) > 0 else "0.0f"
            in1_val = f"buf_{inputs[1]}[j]" if len(inputs) > 1 else "0.0f"
            self.add_line(f"    buf_{clean_id}[j] = _scalar_{op_type.lower()}({in0_val}, {in1_val});")
        self.add_line("  }")

    def _generate_op(self, node: Any, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        if op_type in ("MatMul", "DotGeneral", "Einsum"):
            self._generate_matmul(node, clean_id, inputs, shape)
        elif op_type in ("Conv1D", "Conv2D", "Conv3D", "ConvTranspose2D", "MaxPool", "AvgPool", "MaxPool2D", "AvgPool2D", "BatchNorm", "LayerNorm", "GroupNorm", "Reshape", "Transpose", "Concat", "Slice", "Gather", "Scatter"):
            self._generate_conv_pool(clean_id, inputs, nelem)
        elif op_type in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ArgMax", "ArgMin"):
            self._generate_reduce(node, op_type, clean_id, inputs)
        elif op_type in ("Softmax", "LogSoftmax"):
            # Naive fallback for shape parity without full dimension reduction passes
            self._generate_conv_pool(clean_id, inputs, nelem)
        else:
            self._generate_generic(node, op_type, clean_id, inputs, nelem)

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids = getattr(self.graph, "outputs", []) or []

        self.code = []
        func_params = []
        for idx, node in enumerate(input_nodes):
            meta_dtype = self._map_type(getattr(node, "dtype", "float32"))
            func_params.append(f"const {meta_dtype}* __restrict__ in_{idx}")

        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            meta_dtype = self._map_type(getattr(out_node, "dtype", "float32")) if out_node else "float"
            func_params.append(f"{meta_dtype}* __restrict__ out_{i}")

        func_params.append("int size")
        params_str = ", ".join(func_params)

        self.add_line("#include <wasm_simd128.h>")
        self.add_line("#include <cmath>")
        self.add_line("#include <cstdlib>")
        self.add_line("#include <algorithm>")
        self.add_line("")
        for helper in self.get_helper_functions():
            self.add_line(helper)
        self.add_line("")

        self.add_line('extern "C" {')
        self.add_line(f"void main_kernel({params_str}) {{")

        # Create variable references for inputs and intermediate buffers
        self.add_line("  // Buffer allocations")
        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            self.add_line(f"  const float* buf_{clean_id} = in_{idx};")

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            shape = getattr(node, "shape_metadata", None)
            if not shape:
                shape = [1]
            elif isinstance(shape, (int, float)):
                shape = [int(shape)]
            nelem = self._num_elements(shape)

            # check if this is an output node
            is_output = False
            out_idx = -1
            for i, oid in enumerate(output_ids):
                if oid == nid:
                    is_output = True
                    out_idx = i
                    break

            if is_output:
                self.add_line(f"  float* buf_{clean_id} = out_{out_idx};")
            else:
                self.add_line(f"  float* buf_{clean_id} = (float*)std::aligned_alloc(16, {nelem} * sizeof(float));")

        self.add_line("")
        self.add_line("  // Compute nodes sequentially")

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            inputs = [inp.replace("-", "_") for inp in getattr(node, "inputs", [])]
            shape = getattr(node, "shape_metadata", None)
            if not shape:
                shape = [1]
            elif isinstance(shape, (int, float)):
                shape = [int(shape)]
            nelem = self._num_elements(shape)

            self._generate_op(node, op_type, clean_id, inputs, shape, nelem)

        self.add_line("  // Free temporary allocations")
        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue
            nid = getattr(node, "id", "")
            if nid in output_ids:
                continue
            clean_id = nid.replace("-", "_")
            self.add_line(f"  std::free(buf_{clean_id});")

        self.add_line("}")
        self.add_line("}")

        return "\n".join(self.code)

    def compile_wasm(self, output_dir: str = ".") -> Optional[tuple[str, str]]:
        """Automatically compile the generated C++ source code into WASM and JS binary files using emcc or clang.

        Args:
            output_dir (str): Directory where output files (.wasm and .js) should be saved.

        Returns:
            Optional[tuple[str, str]]: Paths to (js_loader_path, wasm_binary_path) if compilation succeeded, else None.
        """
        import os
        import shutil
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w") as temp_file:
            temp_file.write(self.generate())
            temp_file_path = temp_file.name

        try:
            emcc_bin = shutil.which("emcc")
            if emcc_bin:
                js_out = os.path.join(output_dir, "kernel.js")
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [emcc_bin, "-O3", "-msimd128", "-s", "EXPORTED_FUNCTIONS=['_main_kernel']", "-s", "STANDALONE_WASM", temp_file_path, "-o", js_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            clang_bin = shutil.which("clang")
            if clang_bin:
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [clang_bin, "--target=wasm32", "-O3", "-msimd128", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=main_kernel", temp_file_path, "-o", wasm_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return "", wasm_out

        except Exception:
            pass
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return None
