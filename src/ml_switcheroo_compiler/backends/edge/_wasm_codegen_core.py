"""Intrinsic template resolution, arena layout, and C++ kernel assembly for WASM."""

from __future__ import annotations

import os

import yaml
from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    WasmAttrType,
    _node_attrs,
    _str_list_attr,
)
from ml_switcheroo_compiler.backends.edge._wasm_math_visitors import _WasmMathVisitors
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.errors import UnimplementedMathError


class _WasmCodeGenCore(_WasmMathVisitors):
    """WASM C++ kernel assembly, op dispatch, and memory arena generation layer."""

    @staticmethod
    def _is_wasm_intrinsic_op(op_type: str) -> bool:
        """Check whether an operation is defined in the WASM SIMD intrinsics configuration.

        Args:
            op_type (str): Operation type name.

        Returns:
            bool: True if op_type has an intrinsic or scalar entry in intrinsics.yaml.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if not os.path.exists(yaml_path):
            return False
        with open(yaml_path) as f:
            idata = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
        intrinsics_map = idata.get("intrinsics", {})
        scalars_map = idata.get("scalars", {})
        lower_op = op_type.lower()
        return op_type in intrinsics_map or lower_op in scalars_map or lower_op in [k.lower() for k in intrinsics_map]

    def _emit_mapped_wasm_template(
        self,
        node: IRNode,
        op_type: str,
        clean_id: str,
        inputs: list[str],
        shape: list[int],
        nelem: int,
        mapping: dict[str, WasmAttrType],
    ) -> None:
        """Format and emit a mapped WASM SIMD template for a node.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type name.
            clean_id (str): Sanitized output buffer identifier.
            inputs (list[str]): Sanitized input identifiers.
            shape (list[int]): Output shape dimensions.
            nelem (int): Total number of output elements.
            mapping (dict[str, WasmAttrType]): Registry mapping specification for edge_wasm_simd.

        Raises:
            UnimplementedMathError: If `_emit_mapped_wasm_template` encounters an unsupported operation or missing
                template.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template(str(mapping["template"]))
        if "body" not in template:
            print(f"DEBUG: op_type={op_type}, mapping={mapping}, template={template}")
            raise UnimplementedMathError(f"MISSING BODY FOR: {op_type} template: {template}")

        in0_node: WasmAttrType = (
            next(
                (
                    n
                    for n in self.sorted_nodes
                    if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                ),
                None,
            )
            if inputs
            else None
        )
        raw_in0_shape: BackendArg = (
            adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else None
        )
        in0_shape = self._normalize_ndim_shape(raw_in0_shape, 2)
        shape_seq: list[int] = self._normalize_ndim_shape(shape, 1)

        K = in0_shape[1] if len(in0_shape) > 1 else 1
        N = shape_seq[1] if len(shape_seq) > 1 else 1
        M = shape_seq[0] if len(shape_seq) > 0 else 1

        raw_nelem_in = adapt_backend_module(node).get_symbol("inputs_nelem", [1])
        nelem_in_val = (
            int(coerce_int(raw_nelem_in[0])) if isinstance(raw_nelem_in, (list, tuple)) and raw_nelem_in else 1
        )
        expr_format_args: dict[str, WasmAttrType] = {
            "nelem": nelem,
            "clean_id": clean_id,
            "op_type": op_type,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
            "K": K,
            "N": N,
            "M": M,
            "nelem_in": nelem_in_val,
        }
        expr_format_args.update(mapping)

        body: str = template["body"].format(**expr_format_args)
        for line in body.split("\n"):
            if line.strip():
                self.add_line(f"  {line}")

    def _generate_op(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Dispatch an IR node to its visitor method or WASM SIMD template emitter.

        Args:
            node (IRNode): The IR node to emit.
            op_type (str): Canonical operation type name.
            clean_id (str): Sanitized output buffer variable identifier.
            inputs (list[str]): Sanitized input variable identifiers.
            shape (list[int]): Output tensor shape dimensions.
            nelem (int): Total number of output elements.

        Raises:
            UnimplementedMathError: If `_generate_op` encounters an unsupported operation or missing template.
        """
        if adapt_backend_module(self).has_symbol(f"visit_{op_type}"):
            adapt_backend_module(self).require_op(f"visit_{op_type}")(node, op_type, clean_id, inputs, shape, nelem)
            return

        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import OFFLINE_DIAGNOSTIC_OPS

        if op_type in OFFLINE_DIAGNOSTIC_OPS:
            return

        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op_type, {})
        raw_variants = op_def.get("variants", {})
        raw_wasm = raw_variants.get("edge_wasm_simd", {}) if isinstance(raw_variants, dict) else {}
        mapping: dict[str, WasmAttrType] = dict(raw_wasm) if isinstance(raw_wasm, dict) else {}

        if not mapping or "template" not in mapping:
            if self._is_wasm_intrinsic_op(op_type):
                if len(inputs) <= 1:
                    self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)
                else:
                    self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)
                return
            raise UnimplementedMathError(f"Missing WASM SIMD template for {op_type}")

        self._emit_mapped_wasm_template(node, op_type, clean_id, inputs, shape, nelem, mapping)

    def _build_kernel_signature_params(self, input_nodes: list[IRNode], output_ids: list[str]) -> str:
        """Construct the C++ kernel function parameter declaration string.

        Args:
            input_nodes (list[IRNode]): Graph input IR nodes.
            output_ids (list[str]): Graph output node IDs.

        Returns:
            str: Comma-separated C++ parameter declaration string.
        """
        func_params: list[str] = []
        for idx, node in enumerate(input_nodes):
            meta_dtype: str = self._map_type(coerce_str(adapt_backend_module(node).get_symbol("dtype", "float32")))
            func_params.append(f"const {meta_dtype}* __restrict__ in_{idx}")

        for i, out_id in enumerate(output_ids):
            out_node: WasmAttrType = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_id), None
            )
            meta_dtype = (
                self._map_type(coerce_str(adapt_backend_module(out_node).get_symbol("dtype", "float32")))
                if out_node
                else "float"
            )
            func_params.append(f"{meta_dtype}* __restrict__ out_{i}")

        func_params.append("int size")
        return ", ".join(func_params)

    def _emit_arena_allocations_and_pointers(self, input_nodes: list[IRNode]) -> dict[int, int]:
        """Emit buffer arena allocations, input copies, and node pointer declarations.

        Args:
            input_nodes (list[IRNode]): Graph input IR nodes.

        Returns:
            dict[int, int]: Mapping of arena IDs to allocated byte sizes.
        """
        self.add_line("  // Buffer arenas")
        arenas: dict[int, int] = {}
        for node in self.sorted_nodes:
            attrs = _node_attrs(node)
            arena_id: int = int(coerce_int(attrs.get("buffer_id", 0)))
            offset: int = int(coerce_int(attrs.get("buffer_offset", 0)))
            shape = adapt_backend_module(node).get_symbol("shape_metadata", None)
            shape_seq = self._normalize_ndim_shape(shape, 1)
            nelem: int = self._num_elements(shape_seq)
            arenas[arena_id] = max(arenas.get(arena_id, 0), offset + nelem * 4)

        for arena_id, total_size in arenas.items():
            self.add_line(f"  float* buf_arena_{arena_id} = (float*)std::aligned_alloc(16, {total_size});")

        self.add_line("  // Pointers and Input Copies")
        for idx, node in enumerate(input_nodes):
            clean_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "")).replace("-", "_")
            attrs = _node_attrs(node)
            arena_id = int(coerce_int(attrs.get("buffer_id", 0)))
            offset = int(coerce_int(attrs.get("buffer_offset", 0))) // 4
            shape = adapt_backend_module(node).get_symbol("shape_metadata", None)
            shape_seq = self._normalize_ndim_shape(shape, 1)
            nelem = self._num_elements(shape_seq)
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")
            self.add_line(f"  std::copy(in_{idx}, in_{idx} + {nelem}, buf_{clean_id});")

        for node in self.sorted_nodes:
            if coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) == "Input":
                continue
            clean_id = coerce_str(adapt_backend_module(node).get_symbol("id", "")).replace("-", "_")
            attrs = _node_attrs(node)
            arena_id = int(coerce_int(attrs.get("buffer_id", 0)))
            offset = int(coerce_int(attrs.get("buffer_offset", 0))) // 4
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")

        self.add_line("  float dummy_val = 0.0f;")
        self.add_line("  float* buf_dummy = &dummy_val;")
        self.add_line("")
        return arenas

    def _emit_computed_nodes_and_epilogue(self, output_ids: list[str], arenas: dict[int, int]) -> None:
        """Emit sequential node computations, output copies, and arena deallocations.

        Args:
            output_ids (list[str]): Graph output node IDs.
            arenas (dict[int, int]): Allocated arena IDs to free.
        """
        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import OFFLINE_DIAGNOSTIC_OPS

        self.add_line("  // Compute nodes sequentially")
        for node in self.sorted_nodes:
            op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
            if op_type in ("Input", "Output", "Constant") or op_type in OFFLINE_DIAGNOSTIC_OPS:
                continue

            clean_id = coerce_str(adapt_backend_module(node).get_symbol("id", "")).replace("-", "_")
            inputs: list[str] = [inp.replace("-", "_") for inp in _str_list_attr(node, "inputs")]
            raw_shape = adapt_backend_module(node).get_symbol("shape_metadata", None)
            shape = self._normalize_ndim_shape(raw_shape, 1)
            self._generate_op(node, op_type, clean_id, inputs, shape, self._num_elements(shape))

        self.add_line("  // Copy Outputs")
        for i, out_id in enumerate(output_ids):
            out_node = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_id), None
            )
            if out_node:
                clean_id = out_node.id.replace("-", "_")
                out_shape = adapt_backend_module(out_node).get_symbol("shape_metadata", None)
                out_shape_seq = self._normalize_ndim_shape(out_shape, 1)
                nelem = self._num_elements(out_shape_seq)
                self.add_line(f"  std::copy(buf_{clean_id}, buf_{clean_id} + {nelem}, out_{i});")

        self.add_line("  // Free temporary allocations")
        for arena_id in arenas:
            self.add_line(f"  std::free(buf_arena_{arena_id});")

        self.add_line("}")
        self.add_line("}")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        input_nodes: list[IRNode] = [
            n for n in self.sorted_nodes if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input"
        ]
        output_ids: list[str] = _str_list_attr(self.graph, "outputs")

        self.code.clear()
        params_str: str = self._build_kernel_signature_params(input_nodes, output_ids)

        headers_tpl: str = get_wasm_template("kernel_headers").get("body", "")
        for line in headers_tpl.strip().split("\n"):
            self.add_line(line)
        self.add_line("")
        for helper in self.get_helper_functions():
            self.add_line(helper)
        self.add_line("")

        main_start_tpl: str = get_wasm_template("kernel_main_start").get("body", "")
        for line in main_start_tpl.format(params_str=params_str).strip().split("\n"):
            self.add_line(line)

        arenas = self._emit_arena_allocations_and_pointers(input_nodes)
        self._emit_computed_nodes_and_epilogue(output_ids, arenas)

        return "\n".join(self.code)
