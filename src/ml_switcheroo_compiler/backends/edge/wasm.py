# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
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
        """_allocate_aligned_memory function.

        Args:
        self (Any): The self parameter.
        size_bytes (Any): The size_bytes parameter.
        alignment (Any): The alignment parameter.

        Returns:
        Any: Result.
        """
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        """_generate_striding_logic function.

        Args:
        self (Any): The self parameter.
        shape (Any): The shape parameter.

        Returns:
        Any: Result.
        """
        if not shape:
            return [], "0"
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code = " + ".join([f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)])
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        """_map_type function.

        Args:
        self (Any): The self parameter.
        dtype (Any): The dtype parameter.

        Returns:
        Any: Result.
        """
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "bool": "bool",
        }.get(str(dtype).lower(), "float")

    def _num_elements(self, shape: list[int]) -> int:
        """_num_elements function.

        Args:
        self (Any): The self parameter.
        shape (Any): The shape parameter.

        Returns:
        Any: Result.
        """
        if isinstance(shape, (int, float)):
            return 1
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

    def visit_WhileLoop(self, node: Any, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate WhileLoop."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template("while_loop")
        attrs = getattr(node, "attributes", {})
        body = template["body"].format(clean_id=clean_id, condition_expr=f"buf_{inputs[0]}[0] > 0.0" if inputs else "1", max_iters=attrs.get("max_iters", 10), loop_body=f"// dummy loop body\nbuf_{clean_id}[0] = buf_{inputs[0]}[0];" if inputs else "")
        for line in body.split("\n"):
            self.add_line(line)

    def visit_Cond(self, node: Any, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Cond."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template("cond")
        body = template["body"].format(clean_id=clean_id, condition_expr=f"buf_{inputs[0]}[0] > 0.0" if inputs else "1", true_body=f"// true body\nbuf_{clean_id}[0] = 1.0;", false_body=f"// false body\nbuf_{clean_id}[0] = 0.0;")
        for line in body.split("\n"):
            self.add_line(line)

    def visit_Scan(self, node: Any, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Scan."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template("scan")
        body = template["body"].format(clean_id=clean_id, nelem=nelem, init_val="0.0", scan_op_expr=f"acc_{clean_id} + buf_{inputs[0]}[i]" if inputs else "1.0")
        for line in body.split("\n"):
            self.add_line(line)

    def _generate_op(self, node: Any, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """_generate_op function.

        Args:
        self (Any): The self parameter.
        node (Any): The node parameter.
        op_type (Any): The op_type parameter.
        clean_id (Any): The clean_id parameter.
        inputs (Any): The inputs parameter.
        shape (Any): The shape parameter.
        nelem (Any): The nelem parameter.

        Returns:
        Any: Result.
        """
        if hasattr(self, f"visit_{op_type}"):
            getattr(self, f"visit_{op_type}")(node, op_type, clean_id, inputs, shape, nelem)
            return

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op_type, {})
        mapping = op_def.get("variants", {}).get("edge_wasm_simd", {})

        if not mapping:
            raise UnimplementedMathError(f"Missing WASM SIMD template for {op_type}")
        else:
            print("mapping:", mapping)
            template = get_wasm_template(mapping["template"])
            print("WASM_TEMPLATES keys:", globals().get("_WASM_TEMPLATES", {}).keys() if "_WASM_TEMPLATES" in globals() else "none")
            print("TEMPLATE RETURNED FROM GET_WASM_TEMPLATE:", template)

            # Get dimensions
            in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            if not in0_shape:
                in0_shape = [1, 1]
            elif isinstance(in0_shape, (int, float)):
                in0_shape = [int(in0_shape)]

            K = in0_shape[1] if len(in0_shape) > 1 else 1
            N = shape[1] if len(shape) > 1 else 1
            M = shape[0] if len(shape) > 0 else 1

            expr_format_args = {"nelem": nelem, "clean_id": clean_id, "op_type": op_type, "in0": inputs[0] if len(inputs) > 0 else "dummy", "in1": inputs[1] if len(inputs) > 1 else "dummy", "K": K, "N": N, "M": M, "nelem_in": getattr(node, "inputs_nelem", [1])[0]}
            expr_format_args.update(mapping)

            if "body" not in template:
                raise UnimplementedMathError(f"MISSING BODY FOR: {op_type} template: {template}")
            body = template["body"].format(**expr_format_args)
            for line in body.split("\n"):
                if line.strip():  # pragma: no branch
                    self.add_line(f"  {line}")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids = getattr(self.graph, "outputs", []) or []

        self.code.clear()
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

        self.add_line("  // Buffer arenas")
        arenas = {}
        for node in self.sorted_nodes:
            arena_id = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset = getattr(node, "attributes", {}).get("buffer_offset", 0)
            shape = getattr(node, "shape_metadata", None)
            nelem = self._num_elements(shape if shape else [1])
            if arena_id not in arenas:
                arenas[arena_id] = 0
            arenas[arena_id] = max(arenas[arena_id], offset + nelem * 4)

        for arena_id, total_size in arenas.items():
            self.add_line(f"  float* buf_arena_{arena_id} = (float*)std::aligned_alloc(16, {total_size});")

        self.add_line("  // Pointers and Input Copies")
        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            arena_id = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            shape = getattr(node, "shape_metadata", None)
            nelem = self._num_elements(shape if shape else [1])
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")
            self.add_line(f"  std::copy(in_{idx}, in_{idx} + {nelem}, buf_{clean_id});")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                continue
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            arena_id = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")

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

        self.add_line("  // Copy Outputs")
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            if out_node:
                clean_id = out_node.id.replace("-", "_")
                shape = getattr(out_node, "shape_metadata", None)
                nelem = self._num_elements(shape if shape else [1])
                self.add_line(f"  std::copy(buf_{clean_id}, buf_{clean_id} + {nelem}, out_{i});")

        self.add_line("  // Free temporary allocations")
        for arena_id in arenas:
            self.add_line(f"  std::free(buf_arena_{arena_id});")

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
