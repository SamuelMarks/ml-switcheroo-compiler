# ruff: noqa: E501, C901, PLR0912, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

import uuid
from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class WasmCodeGenerator(BaseGenerator):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated variable names.
        body_lines_simd (list[str]): Generated SIMD execution body lines.
        body_lines_scalar (list[str]): Generated scalar remainder execution body lines.
        is_simd (bool): Internal flag toggling generation mode between SIMD and scalar.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
        """Initialize WasmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines_simd: list[str] = []
        self.body_lines_scalar: list[str] = []
        self.is_simd: bool = False

    def _map_type(self, dtype: str) -> str:
        """Map data type to C++ primitive.

        Args:
            dtype (str): The data type.

        Returns:
            str: C++ primitive type representation.
        """
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "bool": "bool",
        }.get(str(dtype).lower(), "float")

    def _simd_constant(self, res_var: str, val: float, in_vars: list[str]) -> str:
        return f"    v128_t {res_var} = wasm_f32x4_splat({val});"

    def _simd_binary(self, res_var: str, func: str, in_vars: list[str]) -> str:
        return f"    v128_t {res_var} = {func}({in_vars[0]}, {in_vars[1]});"

    def _simd_unary(self, res_var: str, func: str, in_vars: list[str]) -> str:
        return f"    v128_t {res_var} = {func}({in_vars[0]});"

    def _simd_math(self, res_var: str, func: str, in_vars: list[str]) -> str:
        ext = [f"std::{func}(wasm_f32x4_extract_lane({in_vars[0]}, {i}))" for i in range(4)]
        return f"    v128_t {res_var} = wasm_f32x4_make({', '.join(ext)});"

    def _simd_fallback(self, res_var: str, op_type: str, in_vars: list[str]) -> str:
        ext = [f"{op_type.lower()}({', '.join(f'wasm_f32x4_extract_lane({v}, {i})' for v in in_vars)})" for i in range(4)]
        return f"    v128_t {res_var} = wasm_f32x4_make({', '.join(ext)});"

    def _visit_simd(self, node: object, nid: str, op_type: str, res_var_base: str) -> str:
        res_var = f"{res_var_base}_simd"
        self.var_map[nid] = res_var

        if op_type == "Constant":
            self.body_lines_simd.append(self._simd_constant(res_var, node.attributes.get("value", 0.0), []))
            return res_var

        in_vars = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]

        binary_ops = {"Add": "wasm_f32x4_add", "Subtract": "wasm_f32x4_sub", "Multiply": "wasm_f32x4_mul", "TrueDivide": "wasm_f32x4_div", "Div": "wasm_f32x4_div", "Min": "wasm_f32x4_pmin", "Max": "wasm_f32x4_pmax"}
        unary_ops = {"Sqrt": "wasm_f32x4_sqrt", "Abs": "wasm_f32x4_abs", "Negative": "wasm_f32x4_neg", "Neg": "wasm_f32x4_neg"}

        if op_type in binary_ops:
            self.body_lines_simd.append(self._simd_binary(res_var, binary_ops[op_type], in_vars))
        elif op_type in unary_ops:
            self.body_lines_simd.append(self._simd_unary(res_var, unary_ops[op_type], in_vars))
        elif op_type in ("Exp", "Log"):
            self.body_lines_simd.append(self._simd_math(res_var, op_type.lower(), in_vars))
        elif op_type != "Output":
            self.body_lines_simd.append(self._simd_fallback(res_var, op_type, in_vars))

        return res_var

    def _scalar_constant(self, res_var: str, val: float, dtype_c: str) -> str:
        return f"    {dtype_c} {res_var} = {val};"

    def _scalar_binary(self, res_var: str, op: str, in_vars: list[str], dtype_c: str) -> str:
        return f"    {dtype_c} {res_var} = {f' {op} '.join(in_vars)};"

    def _scalar_math(self, res_var: str, func: str, in_vars: list[str], dtype_c: str) -> str:
        return f"    {dtype_c} {res_var} = std::{func}({in_vars[0]});"

    def _visit_scalar(self, node: object, nid: str, op_type: str, res_var_base: str) -> str:
        res_var = f"{res_var_base}_scalar"
        self.var_map[nid] = res_var
        dtype_c = self._map_type(getattr(node, "dtype", "float32"))

        if op_type == "Constant":
            self.body_lines_scalar.append(self._scalar_constant(res_var, node.attributes.get("value", 0.0), dtype_c))
            return res_var

        in_vars = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]
        op_map = {"Add": "+", "Subtract": "-", "Multiply": "*", "TrueDivide": "/", "Div": "/"}

        if op_type in op_map:
            self.body_lines_scalar.append(self._scalar_binary(res_var, op_map[op_type], in_vars, dtype_c))
        elif op_type in ("Exp", "Log", "Sqrt", "Abs"):
            self.body_lines_scalar.append(self._scalar_math(res_var, op_type.lower(), in_vars, dtype_c))
        elif op_type in ("Min", "Max"):
            self.body_lines_scalar.append(f"    {dtype_c} {res_var} = std::{op_type.lower()}({in_vars[0]}, {in_vars[1]});")
        elif op_type in ("Negative", "Neg"):
            self.body_lines_scalar.append(f"    {dtype_c} {res_var} = -{in_vars[0]};")
        elif op_type != "Output":
            self.body_lines_scalar.append(f"    {dtype_c} {res_var} = {op_type.lower()}({', '.join(in_vars)});")

        return res_var

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return ""

        op_type = getattr(node, "op_type", "")
        nid = getattr(node, "id", str(uuid.uuid4()))
        res_var_base = f"v_{nid.replace('-', '_')}"

        if self.is_simd:
            return self._visit_simd(node, nid, op_type, res_var_base)
        else:
            return self._visit_scalar(node, nid, op_type, res_var_base)

    def _generate_simd_loop(self, input_nodes: list, output_ids: list) -> None:
        self.is_simd = True
        self.var_map = self.var_map_simd
        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", "")
            res_var = f"v_{nid.replace('-', '_')}_simd"
            self.var_map_simd[nid] = res_var
            self.body_lines_simd.append(f"    v128_t {res_var} = wasm_v128_load(&in_{idx}[idx]);")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        self.add_line("  int idx = 0;")
        self.add_line("  for (; idx <= size - 4; idx += 4) {")
        for line in self.body_lines_simd:
            self.add_line(line)

        for i, out_id in enumerate(output_ids):
            res_var = self.var_map_simd.get(out_id, out_id)
            self.add_line(f"    wasm_v128_store(&out_{i}[idx], {res_var});")
        self.add_line("  }")

    def _generate_scalar_loop(self, input_nodes: list, output_ids: list) -> None:
        self.is_simd = False
        self.var_map = self.var_map_scalar
        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", "")
            self.var_map_scalar[nid] = f"in_{idx}[idx]"

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        self.add_line("  for (; idx < size; ++idx) {")
        for line in self.body_lines_scalar:
            self.add_line(line)

        for i, out_id in enumerate(output_ids):
            res_var = self.var_map_scalar.get(out_id, out_id)
            self.add_line(f"    out_{i}[idx] = {res_var};")
        self.add_line("  }")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids = getattr(self.graph, "outputs", []) or []

        self.code = []
        self.body_lines_simd = []
        self.body_lines_scalar = []
        self.var_map_simd = {}
        self.var_map_scalar = {}

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
        self.add_line("")
        self.add_line('extern "C" {')
        self.add_line(f"void main_kernel({params_str}) {{")

        self._generate_simd_loop(input_nodes, output_ids)
        self._generate_scalar_loop(input_nodes, output_ids)

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

        # Write generated code to a temporary C++ file
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w") as temp_file:
            temp_file.write(self.generate())
            temp_file_path = temp_file.name

        try:
            # Look for emcc executable in system PATH
            emcc_bin = shutil.which("emcc")
            if emcc_bin:
                js_out = os.path.join(output_dir, "kernel.js")
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [emcc_bin, "-O3", "-msimd128", "-s", "EXPORTED_FUNCTIONS=['_main_kernel']", "-s", "STANDALONE_WASM", temp_file_path, "-o", js_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            # Fallback to clang with wasm target
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
