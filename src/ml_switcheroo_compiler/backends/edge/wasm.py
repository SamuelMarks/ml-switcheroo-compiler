# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

from typing import Optional, Union, cast

WasmAttrType = Union["IRNode", "IRGraph", dict[str, Union[int, float, str, bool, list, tuple, dict, None]], list, str, int, float, tuple, bool, None]

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class WasmCodeGenerator(BaseGenerator):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code."""

    def __init__(self, graph: IRGraph, delegates: WasmAttrType = None) -> None:
        """Initialize WasmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (WasmAttrType, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.is_simd: bool = False

    def _allocate_aligned_memory(self, size_bytes: int, alignment: int = 16) -> str:
        """_allocate_aligned_memory function.

        Args:
        self: The self parameter.
        size_bytes (int): The size_bytes parameter.
        alignment (int): The alignment parameter.

        Returns:
        str: Result.
        """
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        """_generate_striding_logic function.

        Args:
        self: The self parameter.
        shape (list[int]): The shape parameter.

        Returns:
        tuple[list[int], str]: Result.
        """
        if not shape:
            return [], "0"
        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code: str = " + ".join([f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)])
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        """_map_type function.

        Args:
        self: The self parameter.
        dtype (str): The dtype parameter.

        Returns:
        str: Result.
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
        self: The self parameter.
        shape (list[int]): The shape parameter.

        Returns:
        int: Result.
        """
        if isinstance(shape, (int, float)):
            return 1
        n: int = 1
        for s in shape:
            n *= s
        return n

    def get_helper_functions(self) -> list[str]:
        """Return C++ helper functions/macros for missing WASM math.

        Returns:
            list[str]: Lines of C++ macros/functions.
        """
        helpers: list[str] = ["// --- Scalar Fallback Helpers ---"]
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()

                # Load scalar helpers
                scalars: dict[str, str] = data.get("scalars") or {}
                for name, body in scalars.items():
                    helpers.append(f"inline float _scalar_{name.lower()}(float a, float b) {{ {body} }}")

                helpers.append("// --- SIMD Fast Math Approximations from YAML ---")

                intrinsics: dict[str, dict[str, str]] = data.get("intrinsics", {})
                for _op, intrinsic_data in intrinsics.items():
                    if intrinsic_data.get("macro_name") and intrinsic_data.get("simd_expr"):
                        helpers.append(f"inline v128_t {intrinsic_data['macro_name']}(v128_t x) {{")
                        for line in intrinsic_data["simd_expr"].split("\n"):
                            if line.strip():
                                helpers.append(f"    {line}")
                        helpers.append("}")

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers

        helpers.extend(get_cpp_helpers())
        return helpers

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: WasmAttrType) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (WasmAttrType): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def visit_Conv2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Conv2D WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        # Handle Folded BatchNorm Math Transformation
        if attrs.get("folded_batch_norm"):
            bn_inputs: list[str] = attrs.get("bn_inputs", [])
            if len(bn_inputs) == 4:
                scale, bias, mean, var = [self.var_map.get(inp, inp) for inp in bn_inputs]
                eps: float = float(attrs.get("epsilon", 1e-5))
                # We emit scalar C++ for folding
                self.add_line(f"// Folded BatchNorm for {clean_id}")
                self.add_line(f"float mult_{clean_id} = {scale} / std::sqrt({var} + {eps});")
                # This assumes scalar inputs for simplicity in WASM mapping, or we'd map this as a loop over elements

        if attrs.get("tiling"):
            template: dict[str, str] = get_wasm_template("im2col_conv2d")
        else:
            template = get_wasm_template("conv2d")

        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[WasmAttrType] = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs_list]
        in0_shape: list[int] = getattr(input_nodes[0], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        w_shape: list[int] = getattr(input_nodes[1], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1]

        if not in0_shape:
            in0_shape = [1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [1, 1, 1, int(in0_shape)]
        else:
            in0_shape = list(in0_shape)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if not w_shape:
            w_shape = [1, 1, 1, 1]
        elif isinstance(w_shape, (int, float)):
            w_shape = [1, 1, 1, int(w_shape)]
        else:
            w_shape = list(w_shape)
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if not shape:
            shape = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape = [1, 1, 1, int(shape)]
        else:
            shape = list(shape)
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs = getattr(node, "attributes", {})
        stride: WasmAttrType = attrs.get("stride", 1)
        stride_h: int = stride[0] if isinstance(stride, (tuple, list)) else stride
        stride_w: int = stride[1] if isinstance(stride, (tuple, list)) else stride

        expr_args: dict[str, WasmAttrType] = {
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
        body: str = template["body"].format(**expr_args)

        lines: list[str] = []
        for line in body.split("\n"):
            lines.append(f"    {line}")
        for line in lines:
            self.add_line(line)

    def visit_AllReduce(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllReduce."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllReduce", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllGather(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllGather."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllGather", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllToAll(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllToAll."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllToAll", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_ReduceScatter(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC ReduceScatter."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_WhileLoop(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate WhileLoop."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("while_loop")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty loop body"
        if body_graph:
            subgen = WasmCodeGenerator(body_graph)
            loop_body = subgen.generate()

        condition_expr: str = f"buf_{inputs[0]}[0] > 0.0" if inputs else "1"

        body: str = template["body"].format(in0=inputs[0] if inputs else "dummy", clean_id=clean_id, condition_expr=condition_expr, max_iters=attrs.get("max_iters", 10), loop_body=loop_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_Cond(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Cond with dynamic subgraph lowering."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("cond")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        def _lower_branch(branch_graph: WasmAttrType, default_input: str) -> str:
            """Lower a branch subgraph into C++ strings.

            Args:
                branch_graph (WasmAttrType): The subgraph to lower.
                default_input (str): The default input buffer name.

            Returns:
                str: Generated C++ code string.
            """
            if not branch_graph:
                return f"for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

            subgen = WasmCodeGenerator(branch_graph)

            # Map inputs for the subgraph generator to point to the parent's input buffers
            # Offset by 1 because inputs[0] is the conditional predicate
            input_remap = {}
            for i, in_name in enumerate(getattr(branch_graph, "inputs", [])):
                parent_idx = i + 1 if (i + 1) < len(inputs) else 0
                input_remap[in_name] = inputs[parent_idx] if inputs else "dummy"

            # Overload the input mapping logic internally
            subgen.var_map.update(input_remap)

            out_code = subgen.generate()

            # Final output mapping
            if branch_graph.outputs:
                out_id = subgen.var_map.get(branch_graph.outputs[0], branch_graph.outputs[0].replace("-", "_"))
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{out_id}[i];"
            else:
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

            return out_code

        # Check for branch_graphs array first (standardized layout)
        branch_graphs: list[WasmAttrType] = attrs.get("branch_graphs", [])
        if branch_graphs:
            true_graph: WasmAttrType = branch_graphs[0] if len(branch_graphs) > 0 else None
            false_graph: WasmAttrType = branch_graphs[1] if len(branch_graphs) > 1 else None
        else:
            true_graph = attrs.get("then_branch")
            false_graph = attrs.get("else_branch")

        true_body: str = _lower_branch(true_graph, inputs[1] if len(inputs) > 1 else inputs[0] if inputs else "dummy")
        false_body: str = _lower_branch(false_graph, inputs[2] if len(inputs) > 2 else inputs[0] if inputs else "dummy")

        body: str = template["body"].format(condition_expr=f"buf_{inputs[0]}[0] > 0.0" if inputs else "1", true_body=true_body, false_body=false_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_If(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate If (alias for Cond)."""
        self.visit_Cond(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Scan(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Scan using recursive subgraph execution."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("scan")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty scan body"
        if body_graph:
            subgen = WasmCodeGenerator(body_graph)
            loop_body = subgen.generate()

        scan_op_expr: str = f"buf_{clean_id}[i] = buf_{inputs[0]}[i];\\n    " + loop_body.replace("\\n", "\\n    ") if inputs else f"buf_{clean_id}[i] = 1.0f;"
        body: str = template["body"].format(clean_id=clean_id, nelem=nelem, init_val="0.0", scan_op_expr=scan_op_expr)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_MatMul(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MatMul WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})
        if attrs.get("tiling"):
            template: dict[str, str] = get_wasm_template("tiled_matmul")
            # For simplicity, fetch the heuristic from the environment or use defaults
            TILE_M = 4
            TILE_N = 4
            TILE_K = 4
            in0: str = inputs[0] if len(inputs) > 0 else "dummy"
            in1: str = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: list[int] = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

            body: str = template["body"].format(M=M, N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, in0=in0, in1=in1)
            for line in body.split("\n"):
                self.add_line(line)
        else:
            # fallback
            template = get_wasm_template("matmul")
            in0 = inputs[0] if len(inputs) > 0 else "dummy"
            in1 = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            body = template["body"].format(M=M, N=N, K=K, in0=in0, in1=in1, clean_id=clean_id)
            for line in body.split("\n"):
                self.add_line(line)

    def visit_MaxPool2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MaxPool2D WASM."""
        self._generate_pooling2d(node, clean_id, inputs, shape, "max_pool2d")

    def visit_AvgPool2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate AvgPool2D WASM."""
        self._generate_pooling2d(node, clean_id, inputs, shape, "avg_pool2d")

    def _generate_pooling2d(self, node: IRNode, clean_id: str, inputs: list[str], shape: list[int], template_name: str) -> None:
        """Generate Pooling 2D."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template(template_name)
        attrs = getattr(node, "attributes", {})

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "") == inputs[0]), None) if len(inputs) > 0 else None

        in0_shape = getattr(in0_node, "shape_metadata", [1, 1, 1, 1]) if in0_node else [1, 1, 1, 1]

        if not in0_shape:
            in0_shape = [1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [1, 1, 1, int(in0_shape)]
        else:
            in0_shape = list(in0_shape)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if not shape:
            shape = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape = [1, 1, 1, int(shape)]
        else:
            shape = list(shape)
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        kernel_size = attrs.get("kernel_size", attrs.get("pool_size", 1))
        kernel_h = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        kernel_w = kernel_size[1] if isinstance(kernel_size, (list, tuple)) else kernel_size

        stride = attrs.get("stride", attrs.get("strides", 1))
        stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
        stride_w = stride[1] if isinstance(stride, (list, tuple)) else stride

        expr_args = {
            "B": shape[0],
            "C": shape[1],
            "out_height": shape[2],
            "out_width": shape[3],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
        }
        body = template.get("body", "").format(**expr_args)
        for line in body.split("\\n"):
            self.add_line(f"    {line}")

    def visit_Linear(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Linear (Dense) WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Linear")

        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
        N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        has_bias = "true" if in2 != "dummy" else "false"
        body = template.get("body", "").format(M=M, N=N, K=K, in0=in0, in1=in1, in2=in2, clean_id=clean_id, nelem=nelem, has_bias=has_bias)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Attention(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Attention WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Attention")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        seq_len = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        embed_dim = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, seq_len=seq_len, embed_dim=embed_dim)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_MaxPool(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MaxPool WASM."""
        self.visit_MaxPool2D(node, op_type, clean_id, inputs, shape, nelem)

    def visit_LayerNorm(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate LayerNorm WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("LayerNorm")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"  # scale
        in2 = inputs[2] if len(inputs) > 2 else "dummy"  # bias

        attrs = getattr(node, "attributes", {})
        eps = float(attrs.get("epsilon", 1e-5))

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        rows = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        cols = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, rows=rows, cols=cols, eps=eps)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Trig(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Trig (Sin/Cos) WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Exp(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Exp WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Log(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Log WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Tanh(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Tanh WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sigmoid(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Sigmoid WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_ReduceSum(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate ReduceSum WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("ReduceSum")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def _generate_vector_unrolled_op(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate vector unrolled v128 operations with remainder loop peeling and bounds checking."""
        in_id = inputs[0] if inputs else "dummy"

        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        simd_macro = ""
        scalar_expr = f"std::sin(buf_{in_id}[i_{clean_id}])"  # fallback

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
                intr = data.get("intrinsics", {}).get(op_type)
                if intr:
                    simd_macro = intr.get("macro_name", "")
                    if intr.get("scalar_fallback"):
                        scalar_expr = intr["scalar_fallback"].format(f"buf_{in_id}[i_{clean_id}]")

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

    def _generate_op(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """_generate_op function.

        Args:
        self: The self parameter.
        node (IRNode): The node parameter.
        op_type (str): The op_type parameter.
        clean_id (str): The clean_id parameter.
        inputs (list[str]): The inputs parameter.
        shape (list[int]): The shape parameter.
        nelem (int): The nelem parameter.

        Returns:
        None: Result.
        """
        if hasattr(self, f"visit_{op_type}"):
            getattr(self, f"visit_{op_type}")(node, op_type, clean_id, inputs, shape, nelem)
            return

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template
        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        op_def: dict[str, WasmAttrType] = OPS_REGISTRY.get(op_type, {})
        mapping: dict[str, WasmAttrType] = op_def.get("variants", {}).get("edge_wasm_simd", {})

        if not mapping:
            raise UnimplementedMathError(f"Missing WASM SIMD template for {op_type}")
        else:
            template: dict[str, str] = get_wasm_template(mapping["template"])
            if "body" not in template:
                print(f"DEBUG: op_type={op_type}, mapping={mapping}, template={template}")

            # Get dimensions
            in0_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: list[int] = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            if not in0_shape:
                in0_shape = [1, 1]
            elif isinstance(in0_shape, (int, float)):
                in0_shape = [int(in0_shape)]

            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1

            expr_format_args: dict[str, WasmAttrType] = {
                "nelem": nelem,
                "clean_id": clean_id,
                "op_type": op_type,
                "in0": inputs[0] if len(inputs) > 0 else "dummy",
                "in1": inputs[1] if len(inputs) > 1 else "dummy",
                "K": K,
                "N": N,
                "M": M,
                "nelem_in": getattr(node, "inputs_nelem", [1])[0],
            }
            expr_format_args.update(mapping)

            if "body" not in template:
                raise UnimplementedMathError(f"MISSING BODY FOR: {op_type} template: {template}")
            body: str = template["body"].format(**expr_format_args)
            for line in body.split("\n"):
                if line.strip():
                    self.add_line(f"  {line}")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes: list[IRNode] = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids: list[str] = getattr(self.graph, "outputs", []) or []

        self.code.clear()
        func_params: list[str] = []
        for idx, node in enumerate(input_nodes):
            meta_dtype: str = self._map_type(getattr(node, "dtype", "float32"))
            func_params.append(f"const {meta_dtype}* __restrict__ in_{idx}")

        for i, out_id in enumerate(output_ids):
            out_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            meta_dtype = self._map_type(getattr(out_node, "dtype", "float32")) if out_node else "float"
            func_params.append(f"{meta_dtype}* __restrict__ out_{i}")

        func_params.append("int size")
        params_str: str = ", ".join(func_params)

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        headers_tpl: str = get_wasm_template("kernel_headers").get("body", "")
        for line in headers_tpl.strip().split("\n"):
            self.add_line(line)
        self.add_line("")
        for helper in self.get_helper_functions():
            self.add_line(helper)
        self.add_line("")

        main_start_tpl: str = get_wasm_template("kernel_main_start").get("body", "")
        formatted_start: str = main_start_tpl.format(params_str=params_str)
        for line in formatted_start.strip().split("\n"):
            self.add_line(line)

        self.add_line("  // Buffer arenas")
        arenas: dict[int, int] = {}
        for node in self.sorted_nodes:
            arena_id: int = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: int = getattr(node, "attributes", {}).get("buffer_offset", 0)
            shape: list[int] = getattr(node, "shape_metadata", None)
            nelem: int = self._num_elements(shape if shape else [1])
            if arena_id not in arenas:
                arenas[arena_id] = 0
            arenas[arena_id] = max(arenas[arena_id], offset + nelem * 4)

        for arena_id, total_size in arenas.items():
            self.add_line(f"  float* buf_arena_{arena_id} = (float*)std::aligned_alloc(16, {total_size});")

        self.add_line("  // Pointers and Input Copies")
        for idx, node in enumerate(input_nodes):
            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")
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

        self.add_line("  float dummy_val = 0.0f;")
        self.add_line("  float* buf_dummy = &dummy_val;")
        self.add_line("")
        self.add_line("  // Compute nodes sequentially")

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            inputs: list[str] = [inp.replace("-", "_") for inp in getattr(node, "inputs", [])]
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
            temp_file_path: str = temp_file.name

        try:
            emcc_bin = shutil.which("emcc")
            if emcc_bin:
                js_out: str = os.path.join(output_dir, "kernel.js")
                wasm_out: str = os.path.join(output_dir, "kernel.wasm")
                cmd: list[str] = [emcc_bin, "-O3", "-msimd128", "-s", "EXPORTED_FUNCTIONS=['_main_kernel']", "-s", "STANDALONE_WASM", temp_file_path, "-o", js_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            clang_bin = shutil.which("clang")
            if clang_bin:
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [clang_bin, "--target=wasm32", "-O3", "-msimd128", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=main_kernel", temp_file_path, "-o", wasm_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return "", wasm_out

            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError("Neither emcc nor clang found for WASM compilation.")

        except subprocess.CalledProcessError as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            err_msg: str = e.stderr.decode("utf-8") if e.stderr else str(e)
            raise CompilationError(f"WASM compilation failed: {err_msg}") from e
        except Exception as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"WASM compilation failed with unknown error: {e}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
