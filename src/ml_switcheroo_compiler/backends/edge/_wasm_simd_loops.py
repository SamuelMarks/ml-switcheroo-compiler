"""Vectorized v128 SIMD binary and unary unrolled loop emitters for WASM."""

from __future__ import annotations

import os

import yaml
from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.edge._wasm_nn_pooling import _WasmNNPooling
from ml_switcheroo_compiler.core.errors import UnimplementedMathError


class _WasmSimdLoops(_WasmNNPooling):
    """WASM v128 SIMD loop emission helpers for unary and binary elementwise ops."""

    def _generate_binary_simd_op(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate vector unrolled v128 binary operations with remainder loop peeling and bounds checking.

        Args:
            node (IRNode): The IR node.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier for generated code variables.
            inputs (list[str]): Input variable names.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        in0: str = inputs[0] if len(inputs) > 0 else "dummy"
        in1: str = inputs[1] if len(inputs) > 1 else in0

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        simd_macro: str = ""
        scalar_expr: str = ""

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
            intrinsics_dict = data.get("intrinsics", {})
            intr = intrinsics_dict.get(op_type) or next(
                (v for k, v in intrinsics_dict.items() if k.lower() == op_type.lower()), None
            )
            if intr:
                simd_macro = intr.get("macro_name", "")
                if intr.get("scalar_fallback"):
                    scalar_expr = intr["scalar_fallback"].format(f"buf_{in0}[i_{clean_id}]", f"buf_{in1}[i_{clean_id}]")
            if not scalar_expr:
                scalars_dict = data.get("scalars", {})
                if op_type.lower() in scalars_dict:
                    scalar_expr = f"_scalar_{op_type.lower()}(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])"
        if not scalar_expr:
            binary_scalars: dict[str, str] = {
                "add": f"buf_{in0}[i_{clean_id}] + buf_{in1}[i_{clean_id}]",
                "sub": f"buf_{in0}[i_{clean_id}] - buf_{in1}[i_{clean_id}]",
                "mul": f"buf_{in0}[i_{clean_id}] * buf_{in1}[i_{clean_id}]",
                "div": f"buf_{in0}[i_{clean_id}] / buf_{in1}[i_{clean_id}]",
                "min": f"std::min(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])",
                "max": f"std::max(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])",
            }
            scalar_expr = binary_scalars.get(op_type.lower(), f"buf_{in0}[i_{clean_id}]")

        self.add_line("  // Bounds check")
        self.add_line(f"  if ({nelem} > size) return; // Out of bounds")

        if simd_macro:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} <= {nelem} - 4; i_{clean_id} += 4) {{")
            self.add_line(f"      v128_t vec_a = wasm_v128_load(&buf_{in0}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_b = wasm_v128_load(&buf_{in1}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_out = {simd_macro}(vec_a, vec_b);")
            self.add_line(f"      wasm_v128_store(&buf_{clean_id}[i_{clean_id}], vec_out);")
            self.add_line("  }")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")
        else:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")

    def _generate_vector_unrolled_op(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate vector unrolled v128 operations with remainder loop peeling and bounds checking.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).

        Raises:
            UnimplementedMathError: If `_generate_vector_unrolled_op` encounters an unsupported operation or
                missing template.
        """
        in_id = inputs[0] if inputs else "dummy"

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        simd_macro = ""
        scalar_expr = ""

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
            intrinsics_dict = data.get("intrinsics", {})
            intr = intrinsics_dict.get(op_type) or next(
                (v for k, v in intrinsics_dict.items() if k.lower() == op_type.lower()), None
            )
            if intr:
                simd_macro = intr.get("macro_name", "")
                if intr.get("scalar_fallback"):
                    scalar_expr = intr["scalar_fallback"].format(f"buf_{in_id}[i_{clean_id}]")
            if not scalar_expr:
                scalars_dict = data.get("scalars", {})
                scalar_def = scalars_dict.get(op_type.lower())
                if scalar_def:
                    body_str = scalar_def.replace("return ", "").rstrip(";")
                    scalar_expr = body_str.replace("a", f"buf_{in_id}[i_{clean_id}]")
        else:
            builtins = {
                "sin": "std::sin({0})",
                "cos": "std::cos({0})",
                "exp": "std::exp({0})",
                "log": "std::log({0})",
                "tanh": "std::tanh({0})",
                "sqrt": "std::sqrt({0})",
                "abs": "std::abs({0})",
                "neg": "-{0}",
                "relu": "std::max(0.0f, {0})",
            }
            if op_type.lower() in builtins:
                scalar_expr = builtins[op_type.lower()].format(f"buf_{in_id}[i_{clean_id}]")

        if not simd_macro and not scalar_expr:
            raise UnimplementedMathError(f"Operation '{op_type}' lacks WASM SIMD intrinsics and scalar fallback.")

        if simd_macro and not scalar_expr:
            scalar_expr = f"wasm_f32x4_extract_lane({simd_macro}(wasm_f32x4_splat(buf_{in_id}[i_{clean_id}])), 0)"

        self.add_line("  // Bounds check")
        self.add_line(f"  if ({nelem} > size) return; // Out of bounds")

        if simd_macro:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} <= {nelem} - 4; i_{clean_id} += 4) {{")
            self.add_line(f"      v128_t vec_in = wasm_v128_load(&buf_{in_id}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_out = {simd_macro}(vec_in);")
            self.add_line(f"      wasm_v128_store(&buf_{clean_id}[i_{clean_id}], vec_out);")
            self.add_line("  }")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")
        else:
            # Scalar fallback
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")
