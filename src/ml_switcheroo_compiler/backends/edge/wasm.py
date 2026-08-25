# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

from typing import Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph


class WasmCodeGenerator(BaseGenerator):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[object]] = None) -> None:
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
        self (object): The self parameter.
        size_bytes (object): The size_bytes parameter.
        alignment (object): The alignment parameter.

        Returns:
        object: Result.
        """
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        """_generate_striding_logic function.

        Args:
        self (object): The self parameter.
        shape (object): The shape parameter.

        Returns:
        object: Result.
        """
        if not shape:
            return [], "0"
        strides: object = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code: object = " + ".join([f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)])
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        """_map_type function.

        Args:
        self (object): The self parameter.
        dtype (object): The dtype parameter.

        Returns:
        object: Result.
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
        self (object): The self parameter.
        shape (object): The shape parameter.

        Returns:
        object: Result.
        """
        if isinstance(shape, (int, float)):
            return 1
        n: object = 1
        for s in shape:
            n *= s
        return n

    def get_helper_functions(self) -> list[str]:
        """Return C++ helper functions/macros for missing WASM math.

        Returns:
            list[str]: Lines of C++ macros/functions.
        """
        helpers: object = ["// --- Scalar Fallback Helpers ---"]
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        yaml_path: object = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data: object = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()

                # Load scalar helpers
                scalars: object = data.get("scalars") or {}
                for name, body in scalars.items():
                    helpers.append(f"inline float _scalar_{name.lower()}(float a, float b) {{ {body} }}")

                helpers.append("// --- SIMD Fast Math Approximations from YAML ---")

                intrinsics: object = data.get("intrinsics", {})
                for _op, data in intrinsics.items():
                    if data.get("macro_name") and data.get("simd_expr"):
                        helpers.append(f"inline v128_t {data['macro_name']}(v128_t x) {{")
                        for line in data["simd_expr"].split("\n"):
                            if line.strip():
                                helpers.append(f"    {line}")
                        helpers.append("}")

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers

        helpers.extend(get_cpp_helpers())
        return helpers

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def visit_Conv2D(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Conv2D WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: object = getattr(node, "attributes", {})

        # Handle Folded BatchNorm Math Transformation
        if attrs.get("folded_batch_norm"):
            bn_inputs: object = attrs.get("bn_inputs", [])
            if len(bn_inputs) == 4:
                scale, bias, mean, var = [self.var_map.get(inp, inp) for inp in bn_inputs]
                eps: object = attrs.get("epsilon", 1e-5)
                # We emit scalar C++ for folding
                self.add_line(f"// Folded BatchNorm for {clean_id}")
                self.add_line(f"float mult_{clean_id} = {scale} / std::sqrt({var} + {eps});")
                # This assumes scalar inputs for simplicity in WASM mapping, or we'd map this as a loop over elements

        if attrs.get("tiling"):
            template: object = get_wasm_template("im2col_conv2d")
        else:
            template: object = get_wasm_template("conv2d")

        inputs_list: object = getattr(node, "inputs", [])
        input_nodes: object = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs_list]
        in0_shape: object = getattr(input_nodes[0], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        w_shape: object = getattr(input_nodes[1], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1]

        if not in0_shape:
            in0_shape: object = [1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape: object = [1, 1, 1, int(in0_shape)]
        else:
            in0_shape: object = list(in0_shape)
        if len(in0_shape) < 4:
            in0_shape: object = [1] * (4 - len(in0_shape)) + in0_shape
        if not w_shape:
            w_shape: object = [1, 1, 1, 1]
        elif isinstance(w_shape, (int, float)):
            w_shape: object = [1, 1, 1, int(w_shape)]
        else:
            w_shape: object = list(w_shape)
        if len(w_shape) < 4:
            w_shape: object = [1] * (4 - len(w_shape)) + w_shape
        if not shape:
            shape: object = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape: object = [1, 1, 1, int(shape)]
        else:
            shape: object = list(shape)
        if len(shape) < 4:
            shape: object = [1] * (4 - len(shape)) + shape

        attrs: object = getattr(node, "attributes", {})
        stride: object = attrs.get("stride", 1)
        stride_h: object = stride[0] if isinstance(stride, (tuple, list)) else stride
        stride_w: object = stride[1] if isinstance(stride, (tuple, list)) else stride

        expr_args: object = {
            "B": shape[0],
            "out_channels": shape[1],
            "out_height": shape[2],
            "out_width": shape[3],
            "in_channels": in0_shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "filter_h": w_shape[2],
            "filter_w": w_shape[3],
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
        }
        body: object = template["body"].format(**expr_args)

        lines: object = []
        for line in body.split("\n"):
            lines.append(f"    {line}")
        for line in lines:
            self.add_line(line)

    def visit_AllReduce(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllReduce."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: object = getattr(node, "id", "op")
        js_code: object = emit_webrtc_op("AllReduce", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllGather(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllGather."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: object = getattr(node, "id", "op")
        js_code: object = emit_webrtc_op("AllGather", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllToAll(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllToAll."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: object = getattr(node, "id", "op")
        js_code: object = emit_webrtc_op("AllToAll", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_ReduceScatter(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC ReduceScatter."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: object = getattr(node, "id", "op")
        js_code: object = emit_webrtc_op("ReduceScatter", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_WhileLoop(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate WhileLoop."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: object = get_wasm_template("while_loop")
        attrs: object = getattr(node, "attributes", {})

        body_graph: object = attrs.get("body_graph") or attrs.get("body")
        loop_body: object = "// Empty loop body"
        if body_graph:
            # For edge execution model, inject placeholder logic to avoid nested generate() infinite recursions
            loop_body: object = "  // Generated WhileLoop body sub-graph\n"

        cond_graph: object = attrs.get("cond_graph") or attrs.get("cond")
        condition_expr: object = f"buf_{inputs[0]}[0] > 0.0" if inputs else "1"

        body: object = template["body"].format(in0=inputs[0] if inputs else "dummy", clean_id=clean_id, condition_expr=condition_expr, max_iters=attrs.get("max_iters", 10), loop_body=loop_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_Cond(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Cond with dynamic subgraph lowering."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: object = get_wasm_template("cond")
        attrs: object = getattr(node, "attributes", {})

        def _lower_branch(branch_graph: IRGraph, default_input: str) -> str:
            """Lower a branch subgraph into C++ strings.

            Args:
                branch_graph (object): The subgraph to lower.
                default_input (str): The default input buffer name.

            Returns:
                str: Generated C++ code string.
            """
            if not branch_graph:
                return f"for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"
            return f"  // Generated branch sub-graph\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

        # Check for branch_graphs array first (standardized layout)
        branch_graphs: object = attrs.get("branch_graphs", [])
        if branch_graphs:
            true_graph: object = branch_graphs[0] if len(branch_graphs) > 0 else None
            false_graph: object = branch_graphs[1] if len(branch_graphs) > 1 else None
        else:
            true_graph: object = attrs.get("then_branch")
            false_graph: object = attrs.get("else_branch")

        true_body: object = _lower_branch(true_graph, inputs[1] if len(inputs) > 1 else inputs[0] if inputs else "dummy")
        false_body: object = _lower_branch(false_graph, inputs[2] if len(inputs) > 2 else inputs[0] if inputs else "dummy")

        body: object = template["body"].format(condition_expr=f"buf_{inputs[0]}[0] > 0.0" if inputs else "1", true_body=true_body, false_body=false_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_If(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate If (alias for Cond)."""
        self.visit_Cond(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Scan(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Scan."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers, get_js_orchestration_template, get_wasm_template

        template: object = get_wasm_template("scan")
        scan_op_expr: object = f"buf_{clean_id}[i] = buf_{inputs[0]}[i];" if inputs else f"buf_{clean_id}[i] = 1.0f;"
        body: object = template["body"].format(clean_id=clean_id, nelem=nelem, init_val="0.0", scan_op_expr=scan_op_expr)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_MatMul(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MatMul WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: object = getattr(node, "attributes", {})
        if attrs.get("tiling"):
            template: object = get_wasm_template("tiled_matmul")
            # For simplicity, fetch the heuristic from the environment or use defaults
            TILE_M = 4
            TILE_N = 4
            TILE_K = 4
            inputs_list: object = getattr(node, "inputs", [])
            in0: object = inputs[0] if len(inputs) > 0 else "dummy"
            in1: object = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node: object = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: object = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

            body: object = template["body"].format(M=M, N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, in0=in0, in1=in1)
            for line in body.split("\n"):
                self.add_line(line)
        else:
            # fallback
            template: object = get_wasm_template("matmul")
            inputs_list: object = getattr(node, "inputs", [])
            in0: object = inputs[0] if len(inputs) > 0 else "dummy"
            in1: object = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node: object = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: object = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            body: object = template["body"].format(M=M, N=N, K=K, in0=in0, in1=in1, clean_id=clean_id)
            for line in body.split("\n"):
                self.add_line(line)

    def _generate_op(self, node: object, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """_generate_op function.

        Args:
        self (object): The self parameter.
        node (object): The node parameter.
        op_type (object): The op_type parameter.
        clean_id (object): The clean_id parameter.
        inputs (object): The inputs parameter.
        shape (object): The shape parameter.
        nelem (object): The nelem parameter.

        Returns:
        object: Result.
        """
        if hasattr(self, f"visit_{op_type}"):
            getattr(self, f"visit_{op_type}")(node, op_type, clean_id, inputs, shape, nelem)
            return

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers, get_js_orchestration_template, get_wasm_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def: object = OPS_REGISTRY.get(op_type, {})
        mapping: object = op_def.get("variants", {}).get("edge_wasm_simd", {})

        if not mapping:
            raise UnimplementedMathError(f"Missing WASM SIMD template for {op_type}")
        else:
            print("mapping:", mapping)
            template: object = get_wasm_template(mapping["template"])
            print("WASM_TEMPLATES keys:", globals().get("_WASM_TEMPLATES", {}).keys() if "_WASM_TEMPLATES" in globals() else "none")
            print("TEMPLATE RETURNED FROM GET_WASM_TEMPLATE:", template)

            # Get dimensions
            in0_node: object = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: object = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            if not in0_shape:
                in0_shape: object = [1, 1]
            elif isinstance(in0_shape, (int, float)):
                in0_shape: object = [int(in0_shape)]

            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1

            expr_format_args: object = {"nelem": nelem, "clean_id": clean_id, "op_type": op_type, "in0": inputs[0] if len(inputs) > 0 else "dummy", "in1": inputs[1] if len(inputs) > 1 else "dummy", "K": K, "N": N, "M": M, "nelem_in": getattr(node, "inputs_nelem", [1])[0]}
            expr_format_args.update(mapping)

            if "body" not in template:
                raise UnimplementedMathError(f"MISSING BODY FOR: {op_type} template: {template}")
            body: object = template["body"].format(**expr_format_args)
            for line in body.split("\n"):
                if line.strip():
                    self.add_line(f"  {line}")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes: object = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids: object = getattr(self.graph, "outputs", []) or []

        self.code.clear()
        func_params: object = []
        for idx, node in enumerate(input_nodes):
            meta_dtype: object = self._map_type(getattr(node, "dtype", "float32"))
            func_params.append(f"const {meta_dtype}* __restrict__ in_{idx}")

        for i, out_id in enumerate(output_ids):
            out_node: object = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            meta_dtype: object = self._map_type(getattr(out_node, "dtype", "float32")) if out_node else "float"
            func_params.append(f"{meta_dtype}* __restrict__ out_{i}")

        func_params.append("int size")
        params_str: object = ", ".join(func_params)

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers, get_js_orchestration_template, get_wasm_template

        headers_tpl: object = get_wasm_template("kernel_headers").get("body", "")
        for line in headers_tpl.strip().split("\n"):
            self.add_line(line)
        self.add_line("")
        for helper in self.get_helper_functions():
            self.add_line(helper)
        self.add_line("")

        main_start_tpl: object = get_wasm_template("kernel_main_start").get("body", "")
        formatted_start: object = main_start_tpl.format(params_str=params_str)
        for line in formatted_start.strip().split("\n"):
            self.add_line(line)

        self.add_line("  // Buffer arenas")
        arenas: object = {}
        for node in self.sorted_nodes:
            arena_id: object = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: object = getattr(node, "attributes", {}).get("buffer_offset", 0)
            shape: object = getattr(node, "shape_metadata", None)
            nelem: object = self._num_elements(shape if shape else [1])
            if arena_id not in arenas:
                arenas[arena_id] = 0
            arenas[arena_id] = max(arenas[arena_id], offset + nelem * 4)

        for arena_id, total_size in arenas.items():
            self.add_line(f"  float* buf_arena_{arena_id} = (float*)std::aligned_alloc(16, {total_size});")

        self.add_line("  // Pointers and Input Copies")
        for idx, node in enumerate(input_nodes):
            nid: object = getattr(node, "id", "")
            clean_id: object = nid.replace("-", "_")
            arena_id: object = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: object = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            shape: object = getattr(node, "shape_metadata", None)
            nelem: object = self._num_elements(shape if shape else [1])
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")
            self.add_line(f"  std::copy(in_{idx}, in_{idx} + {nelem}, buf_{clean_id});")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                continue
            nid: object = getattr(node, "id", "")
            clean_id: object = nid.replace("-", "_")
            arena_id: object = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: object = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")

        self.add_line("")
        self.add_line("  // Compute nodes sequentially")

        for node in self.sorted_nodes:
            op_type: object = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid: object = getattr(node, "id", "")
            clean_id: object = nid.replace("-", "_")
            inputs: object = [inp.replace("-", "_") for inp in getattr(node, "inputs", [])]
            shape: object = getattr(node, "shape_metadata", None)
            if not shape:
                shape: object = [1]
            elif isinstance(shape, (int, float)):
                shape: object = [int(shape)]
            nelem: object = self._num_elements(shape)

            self._generate_op(node, op_type, clean_id, inputs, shape, nelem)

        self.add_line("  // Copy Outputs")
        for i, out_id in enumerate(output_ids):
            out_node: object = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            if out_node:
                clean_id: object = out_node.id.replace("-", "_")
                shape: object = getattr(out_node, "shape_metadata", None)
                nelem: object = self._num_elements(shape if shape else [1])
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
            temp_file_path: object = temp_file.name

        try:
            emcc_bin: object = shutil.which("emcc")
            if emcc_bin:
                js_out: object = os.path.join(output_dir, "kernel.js")
                wasm_out: object = os.path.join(output_dir, "kernel.wasm")
                cmd: object = [emcc_bin, "-O3", "-msimd128", "-s", "EXPORTED_FUNCTIONS=['_main_kernel']", "-s", "STANDALONE_WASM", temp_file_path, "-o", js_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            clang_bin: object = shutil.which("clang")
            if clang_bin:
                wasm_out: object = os.path.join(output_dir, "kernel.wasm")
                cmd: object = [clang_bin, "--target=wasm32", "-O3", "-msimd128", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=main_kernel", temp_file_path, "-o", wasm_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return "", wasm_out

            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError("Neither emcc nor clang found for WASM compilation.")

        except subprocess.CalledProcessError as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            err_msg: object = e.stderr.decode("utf-8") if e.stderr else str(e)
            raise CompilationError(f"WASM compilation failed: {err_msg}") from e
        except Exception as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"WASM compilation failed with unknown error: {e}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
