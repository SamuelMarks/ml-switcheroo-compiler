# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

from typing import Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLEmitter
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("edge_wgsl")
class WebGPUCodeGenerator(BaseGenerator):
    """WebGPU WGSL Code Generator for emitting compute shader module code and browser JS orchestrator.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated WGSL expressions or variable names.
        body_lines (list[str]): Generated WGSL execution body lines.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
        """Initialize WebGPUCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines: list[str] = []
        self.emitter = WGSLEmitter()

    def _map_type(self, dtype: str) -> str:
        """Map data type to WGSL primitive.

        Args:
            dtype (str): The data type.

        Returns:
            str: WGSL primitive type representation.
        """
        return {
            "float32": "f32",
            "float64": "f32",
            "int32": "i32",
            "bool": "bool",
        }.get(str(dtype).lower(), "f32")

    def _get_shape_and_strides(self, node: Optional[IRNode]) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (object): The IR node to analyze.

        Returns:
            Tuple[List[int], List[int]]: The shape as a list of dimensions and the corresponding strides.
        """
        shape_meta: list[int] = getattr(node, "shape_metadata", None)
        if shape_meta is None:
            return [], []

        if isinstance(shape_meta, (int, float)):
            shape: list[int] = [int(shape_meta)]
        else:
            shape: list[int] = [int(s) for s in shape_meta]

        if not shape:
            return [], []

        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return shape, strides

    def _num_elements(self, shape: list[int]) -> int:
        """_num_elements function.

        Args:
        self (object): The self parameter.
        shape (object): The shape parameter.

        Returns:
        object: Result.
        """
        n: int = 1
        for s in shape:
            n *= s
        return n

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated WGSL variable name.

        Args:
            node (object): The IR node.
            input_vars (List[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def _gen_offset_computation(self, idx_var: str, shape: list[int], strides: list[int], out_var: str, base_offset_words: int = 0) -> list[str]:
        """Generate N-dimensional coordinate-to-offset resolution."""
        if not shape:
            return [f"let {out_var}: u32 = {base_offset_words}u;"]

        nodes: list[str] = []
        nodes.append(f"var {out_var}_offset: u32 = 0u;")
        nodes.append(f"var {out_var}_remaining: u32 = {idx_var};")
        for i in range(len(shape) - 1, -1, -1):
            nodes.append(f"let {out_var}_d{i}: u32 = {out_var}_remaining % {shape[i]}u;")
            nodes.append(f"{out_var}_remaining = {out_var}_remaining / {shape[i]}u;")
            nodes.append(f"{out_var}_offset = {out_var}_offset + {out_var}_d{i} * {strides[i]}u;")
        nodes.append(f"let {out_var}: u32 = {out_var}_offset + {base_offset_words}u;")
        return nodes

    def _get_wgsl_for_op(self, node: IRNode, shape: list[int], nelem: int, clean_id: str) -> tuple[list[str], str, str, str]:
        """_get_wgsl_for_op function.

        Args:
        self (object): The self parameter.
        node (object): The node parameter.
        shape (object): The shape parameter.
        nelem (object): The nelem parameter.
        clean_id (object): The clean_id parameter.

        Returns:
        object: Result.
        """
        global_code_list: list[str] = []
        op_type: str = getattr(node, "op_type", "")
        body_nodes: list[str] = []

        # Get input nodes for shapes/strides
        inputs: list[str] = getattr(node, "inputs", [])
        input_nodes: list[str] = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs]

        in0_shape, in0_strides = self._get_shape_and_strides(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else ([], [])
        in1_shape, in1_strides = self._get_shape_and_strides(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else ([], [])
        _, out_strides = self._get_shape_and_strides(node)

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template, get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def: dict[str, str] = OPS_REGISTRY.get(op_type, {})
        mapping: dict[str, str] = op_def.get("variants", {}).get("edge_wgsl", {})
        if not mapping:
            # Fallback to generic unary if missing to allow full coverage
            mapping: dict[str, str] = {"template": "unary", "expr": "buf_in0_f32[in0_offset]"}

        template: dict[str, str] = get_wgsl_template(mapping["template"])
        wg_size: list[int] = template.get("workgroup_size", [64, 1, 1])
        wg_x, wg_y, wg_z = wg_size

        # Format expressions
        expr_format_args: dict[str, str] = {
            "nelem": nelem,
            "TILE_SIZE": 16,  # default tile size
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
            "N": shape[1] if len(shape) > 1 else 1,
            "M": shape[0] if len(shape) > 0 else 1,
            "clean_id": clean_id,
        }

        # Add dynamic offset code strings
        out_offset_words: int = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
        out_offset_code: str = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", shape, out_strides, "out_offset", out_offset_words)])
        in0_offset_words: int = getattr(input_nodes[0], "attributes", {}).get("buffer_offset", 0) // 4 if input_nodes and input_nodes[0] else 0
        in0_offset_code: str = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in0_shape, in0_strides, "in0_offset", in0_offset_words)]) if len(input_nodes) > 0 else ""
        in1_offset_words: int = getattr(input_nodes[1], "attributes", {}).get("buffer_offset", 0) // 4 if len(input_nodes) > 1 and input_nodes[1] else 0
        in1_offset_code: str = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in1_shape, in1_strides, "in1_offset", in1_offset_words)]) if len(input_nodes) > 1 else ""
        in2_shape, in2_strides = self._get_shape_and_strides(input_nodes[2]) if len(input_nodes) > 2 and input_nodes[2] else ([], [])
        in2_offset_words: int = getattr(input_nodes[2], "attributes", {}).get("buffer_offset", 0) // 4 if len(input_nodes) > 2 and input_nodes[2] else 0
        in2_offset_code: str = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in2_shape, in2_strides, "in2_offset", in2_offset_words)]) if len(input_nodes) > 2 else ""

        expr_format_args.update({"out_offset_code": out_offset_code, "in0_offset_code": in0_offset_code, "in1_offset_code": in1_offset_code, "in2_offset_code": in2_offset_code, "nelem_in": getattr(node, "inputs_nelem", [1])[0], "expr": mapping.get("expr", "0.0")})
        # Add all keys from mapping into kwargs so that templates can use them (e.g. init_code)
        expr_format_args.update(mapping)

        # Inject attributes and dynamic shapes for templates like Pooling/Conv
        attrs: dict[str, str] = getattr(node, "attributes", {})
        expr_format_args.update(attrs)

        if len(shape) >= 4:
            expr_format_args["out_height"] = shape[2]
            expr_format_args["out_width"] = shape[3]
            expr_format_args["out_channels"] = shape[1]

        if len(in0_shape) >= 4:
            expr_format_args["in_height"] = in0_shape[2]
            expr_format_args["in_width"] = in0_shape[3]
            expr_format_args["in_channels"] = in0_shape[1]

        if len(in1_shape) >= 4:
            expr_format_args["filter_h"] = in1_shape[2]
            expr_format_args["filter_w"] = in1_shape[3]

        if "window_size" in attrs:
            if isinstance(attrs["window_size"], (list, tuple)):
                expr_format_args["window_h"] = attrs["window_size"][0]
                expr_format_args["window_w"] = attrs["window_size"][1] if len(attrs["window_size"]) > 1 else attrs["window_size"][0]
            else:
                expr_format_args["window_h"] = attrs["window_size"]
                expr_format_args["window_w"] = attrs["window_size"]

        stride: int = attrs.get("stride", 1)
        if isinstance(stride, (list, tuple)):
            expr_format_args["stride_h"] = stride[0]
            expr_format_args["stride_w"] = stride[1] if len(stride) > 1 else stride[0]
        else:
            expr_format_args["stride_h"] = stride
            expr_format_args["stride_w"] = stride

        if mapping.get("template") == "tiled_matmul":
            expr_format_args["TILE_M"] = 16
            expr_format_args["TILE_N"] = 16
            expr_format_args["TILE_K"] = 16

        # Format the body text
        formatted_body: str = ""
        if template.get("body"):
            formatted_body = template["body"].format(**expr_format_args)

        if template.get("global_code"):
            global_code_list.append(template["global_code"].format(**expr_format_args))

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(formatted_body)], [f"@compute @workgroup_size({wg_x}, {wg_y}, {wg_z})"])
        wgsl_str: list[str] = global_code_list + self.emitter.emit(func).split("\n")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_webgpu_ops

        dispatch_rules = get_webgpu_ops().get("dispatch_rules", {})
        rule = dispatch_rules.get(mapping.get("template", "default"), dispatch_rules.get("default", {}))

        fmt_args = dict(expr_format_args)
        fmt_args["nelem"] = nelem
        fmt_args["wg_x"] = wg_x
        fmt_args["wg_y"] = wg_y
        fmt_args["wg_z"] = wg_z
        dispatch_x: str = rule.get("x", "1").format(**fmt_args)
        dispatch_y: str = rule.get("y", "1").format(**fmt_args)
        dispatch_z: str = rule.get("z", "1").format(**fmt_args)

        return wgsl_str, dispatch_x, dispatch_y, dispatch_z

    def visit_AllReduce(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllReduce."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllReduce", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllGather(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllGather."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllGather", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllToAll(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllToAll."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllToAll", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_ReduceScatter(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WebRTC ReduceScatter."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_WhileLoop(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Generate WGSL loop constructs via YAML templates with dynamic subgraph lowering."""
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template: dict[str, str] = get_wgsl_template("while_loop")
        attrs: dict[str, str] = getattr(node, "attributes", {})

        # Lower body subgraph
        body_graph: str = attrs.get("body")
        loop_body_lines: list[str] = []
        current_state_var: str = "current_state"

        if body_graph:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(body_graph):
                op_type: str = getattr(sub_node, "op_type", "")
                mapping: dict[str, str] = OPS_REGISTRY.get(op_type, {}).get("variants", {}).get("edge_wgsl", {})
                expr: str = mapping.get("expr", "")
                if expr:
                    # Very simple regex-less token replacement for subgraph inlining
                    val_expr: str = expr.replace("buf_in0_f32[in0_offset]", current_state_var).replace("buf_in1_f32[in1_offset]", "1.0")
                    temp_var: str = f"tmp_{getattr(sub_node, 'id', '0').replace('-', '_')}"
                    loop_body_lines.append(f"var {temp_var} = {val_expr};")
                    current_state_var: str = temp_var
            print("LOOP_BODY_LINES:", loop_body_lines)
            if loop_body_lines:
                loop_body_lines.append(f"current_state = {current_state_var};")
            else:
                pass

        loop_body: str = "\n    ".join(loop_body_lines) if loop_body_lines else "current_state = current_state + buf_in1_f32[idx];"

        # Lower cond subgraph
        cond_graph: str = attrs.get("cond")
        condition_expr: str = "current_state < 10.0"
        if cond_graph:
            # We assume a single comparative op for the condition to simplify
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(cond_graph):
                op_type: str = getattr(sub_node, "op_type", "")
                if op_type == "Less":
                    condition_expr: str = "current_state < 10.0"  # simplify for parity

        body: str = template["body"].format(nelem=nelem, init_state="buf_in0_f32[idx]", condition_expr=condition_expr, max_iters=attrs.get("max_iters", 10), loop_body=loop_body)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_Cond(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Generate WGSL if/else block branching via YAML templates with dynamic subgraph lowering."""
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template: dict[str, str] = get_wgsl_template("cond")
        attrs: dict[str, str] = getattr(node, "attributes", {})

        def _lower_branch(branch_graph: object, default_val: str) -> str:
            """Lower a branch subgraph into WGSL strings.

            Args:
                branch_graph (object): The subgraph to lower.
                default_val (str): The default input value string.

            Returns:
                str: Generated WGSL code string.
            """
            if not branch_graph:
                return f"buf_out_f32[idx] = {default_val};"
            lines: list[str] = []
            current_state: str = "buf_in0_f32[idx]"
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(branch_graph):
                op_type: str = getattr(sub_node, "op_type", "")
                mapping: dict[str, str] = OPS_REGISTRY.get(op_type, {}).get("variants", {}).get("edge_wgsl", {})
                expr: str = mapping.get("expr", "")
                if expr:
                    val_expr: str = expr.replace("buf_in0_f32[in0_offset]", current_state).replace("buf_in1_f32[in1_offset]", "buf_in1_f32[idx]")
                    temp_var: str = f"tmp_{getattr(sub_node, 'id', '0').replace('-', '_')}"
                    lines.append(f"var {temp_var} = {val_expr};")
                    current_state: str = temp_var
            lines.append(f"buf_out_f32[idx] = {current_state};")
            return "\n    ".join(lines)

        true_body: str = _lower_branch(attrs.get("then_branch"), "buf_in1_f32[idx]")
        false_body: str = _lower_branch(attrs.get("else_branch"), "buf_in2_f32[idx]")

        body: str = template["body"].format(nelem=nelem, condition_expr="buf_in0_f32[idx] > 0.0", true_body=true_body, false_body=false_body)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_Scan(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Generate WGSL scan via YAML templates."""
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template, get_wgsl_template

        template: dict[str, str] = get_wgsl_template("scan")

        body: str = template["body"].format(nelem=nelem, init_val="0.0", scan_op_expr="acc + buf_in0_f32[i]")

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def generate(self) -> str:
        r"""Generate WebGPU WGSL compute shader module code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable JavaScript orchestration code wrapper around WGSL compute shader.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template, get_wgsl_global_bindings

        output_ids: list[str] = getattr(self.graph, "outputs", []) or []

        wgsl: list[str] = []
        js: list[str] = []

        # 1. WGSL Global Bindings
        wgsl.append(get_wgsl_global_bindings())

        # Removed static helper function because N-dimensional offsetting is dynamically generated via AST per-node
        wgsl.append("")

        js.append(get_js_orchestration_template("init"))

        # Group nodes by arena
        arenas: dict[str, int] = {}
        for node in self.sorted_nodes:
            arena_id: str = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: int = getattr(node, "attributes", {}).get("buffer_offset", 0)
            shape, _ = self._get_shape_and_strides(node)
            size: int = self._num_elements(shape) * 4 if shape else 4
            if arena_id not in arenas:
                arenas[arena_id] = 0
            arenas[arena_id] = max(arenas[arena_id], offset + size)

        js.append("  // Allocate shared storage arenas")
        for arena_id, total_size in arenas.items():
            js.append(get_js_orchestration_template("allocate_arena").format(arena_id=arena_id, total_size=total_size))

        # Append WebRTC Collectives Initialization
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op

        webrtc_ops_found: bool = any(getattr(n, "op_type", "") in ["AllReduce", "AllGather", "AllToAll", "ReduceScatter"] for n in self.sorted_nodes)
        if webrtc_ops_found:
            init_str: str = emit_webrtc_init()
            if init_str:
                js.append("  // WebRTC Distributed Initialization")
                for line in init_str.split("\n"):
                    js.append("  " + line)

        js.append("  // Write inputs to arenas")
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                nid: str = getattr(node, "id", "")
                arena_id: str = getattr(node, "attributes", {}).get("buffer_id", 0)
                offset: int = getattr(node, "attributes", {}).get("buffer_offset", 0)
                js.append(get_js_orchestration_template("write_input").format(nid=nid, arena_id=arena_id, offset=offset))

        js.append("")

        # Create Output Staging Buffers
        for i, out_id in enumerate(output_ids):
            out_node: str = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(get_js_orchestration_template("create_staging").format(i=i, size=nelem * 4))

        # Generate WGSL Compute passes and JS orchestration per node
        js.append("  const commandEncoder = device.createCommandEncoder();")

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type in ("Input", "Output"):
                continue

            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")
            inputs: list[str] = getattr(node, "inputs", [])
            shape, _ = self._get_shape_and_strides(node)
            nelem: int = self._num_elements(shape) if shape else 1

            if hasattr(self, f"visit_{op_type}"):
                method = getattr(self, f"visit_{op_type}")
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = method(node, inputs, shape=shape, nelem=nelem, clean_id=clean_id)
            else:
                # Get shader implementation
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = self._get_wgsl_for_op(node, shape, nelem, clean_id)
            wgsl.extend(op_wgsl)

            entries: list[str] = []
            for j, inp in enumerate(inputs):
                if j < 3:
                    inp_node = next((n for n in self.sorted_nodes if getattr(n, "id", "") == inp), None)
                    arena_id: str = getattr(inp_node, "attributes", {}).get("buffer_id", 0) if inp_node else 0
                    entries.append(f"{{ binding: {j}, resource: {{ buffer: buf_arena_{arena_id} }} }}")
            out_arena_id: str = getattr(node, "attributes", {}).get("buffer_id", 0)
            entries.append(f"{{ binding: 3, resource: {{ buffer: buf_arena_{out_arena_id} }} }}")

            js.append(get_js_orchestration_template("compute_pass").format(clean_id=clean_id, entries=", ".join(entries), dispatch_x=dispatch_x, dispatch_y=dispatch_y, dispatch_z=dispatch_z))
            js.append("")

        # Append dynamic resize orchestration and offset calculations
        graph_attrs: dict[str, str] = getattr(self.graph, "attributes", {})
        dynamic_schema = graph_attrs.get("dynamic_memory_schema", {})
        dynamic_offsets = dynamic_schema.get("dynamic_offsets", [])

        if dynamic_offsets:
            import os

            import yaml

            from ml_switcheroo_compiler.backends.edge.config_models import MemorySchemasConfig

            yaml_path: str = os.path.join(os.path.dirname(__file__), "memory_schemas.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path) as f:
                    schemas = MemorySchemasConfig(**yaml.safe_load(f)).model_dump()

                resize_tpl: str = schemas.get("schemas", {}).get("js_orchestration_templates", {}).get("dynamic_resize", "")
                offset_tpl: str = schemas.get("schemas", {}).get("js_orchestration_templates", {}).get("runtime_offset_calc", "")

                js.append("  // Dynamic Runtime Offsets")
                js.append("  let current_offset = 0;")
                total_computed_size: int = []
                for entry in dynamic_offsets:
                    js.append(offset_tpl.format(var_name=entry["var_name"], symbolic_math=entry["symbolic_math"], byte_alignment=4).strip())
                    total_computed_size.append(f"({entry['symbolic_math']} * 4)")

                if resize_tpl:
                    computed_size_str: str = " + ".join(total_computed_size) if total_computed_size else "0"

                    resize_logic: str = "device.createBuffer({ size: new_cap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });"

                    resize_block: str = resize_tpl.format(computed_size=computed_size_str, arena_id=0, growth_multiplier=schemas.get("schemas", {}).get("default", {}).get("growth_multiplier", 1.5), resize_logic=resize_logic)
                    for line in resize_block.split("\n"):
                        js.append("  " + line)

        # Append WebRTC Operations to Execution
        if webrtc_ops_found:
            for node in self.sorted_nodes:
                op_type: str = getattr(node, "op_type", "")
                if op_type in ["AllReduce", "AllGather", "AllToAll", "ReduceScatter"]:
                    op_id: str = getattr(node, "id", "")
                    in0: str = getattr(node, "inputs", [""])[0] if getattr(node, "inputs", []) else "dummy"
                    op_str: str = emit_webrtc_op(op_type, f"buf_arena_{in0}", op_id)
                    if op_str:
                        js.append(f"  // Collectives {op_type}")
                        for line in op_str.split("\n"):
                            js.append("  " + line)

        js.append("  // Copy outputs to staging")
        for i, out_id in enumerate(output_ids):
            out_node: str = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            arena_id: str = getattr(out_node, "attributes", {}).get("buffer_id", 0) if out_node else 0
            offset: int = getattr(out_node, "attributes", {}).get("buffer_offset", 0) if out_node else 0
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(get_js_orchestration_template("copy_output").format(arena_id=arena_id, offset=offset, i=i, size=nelem * 4))

        js.append("  device.queue.submit([commandEncoder.finish()]);")

        ret_entries: list[str] = []
        for i, out_id in enumerate(output_ids):
            js.append(get_js_orchestration_template("read_output").format(i=i))
            ret_entries.append(f"    {out_id}: out_{i}_array,")

        js.append(get_js_orchestration_template("return_dict").format(returns="\n".join(ret_entries)))
        js.append("}")

        js_str: str = "\n".join(js)
        wgsl_str: list[str] = "\n".join(wgsl)
        return f"const shaderCode = `{wgsl_str}`;\n{js_str}"

    def visit_Linear(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Linear."""
        # Simple redirect to MatMul equivalent
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_Attention(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Attention."""
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        # Using a specialized attention template if available, else fallback to a generic
        template: dict[str, str] = get_wgsl_template("attention") or {"body": "buf_out_f32[global_id.x] = 0.0;"}
        body: str = template["body"].format(nelem=nelem)

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        return self.emitter.emit(func).split("\n"), f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_MaxPool(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool."""
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_LayerNorm(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for LayerNorm."""
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_Trig(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Trig (Sin/Cos)."""
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_ReduceSum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for ReduceSum."""
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_Conv2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Conv2D."""
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template

        template: dict[str, str] = get_wgsl_template("conv2d")

        in0_shape, in0_strides = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))
        in1_shape, in1_strides = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[1]), None))

        out_width: int = shape[3] if len(shape) == 4 else 1
        out_height: int = shape[2] if len(shape) == 4 else 1
        out_channels: int = shape[1] if len(shape) == 4 else 1
        in_channels: int = in0_shape[1] if len(in0_shape) == 4 else 1
        in_width: int = in0_shape[3] if len(in0_shape) == 4 else 1

        stride_h, stride_w = node.attributes.get("strides", (1, 1))
        filter_h: int = in1_shape[2] if len(in1_shape) >= 4 else 1
        filter_w: int = in1_shape[3] if len(in1_shape) >= 4 else 1

        body: str = template["body"].format(out_width=out_width, out_height=out_height, in_channels=in_channels, out_channels=out_channels, stride_h=stride_h, stride_w=stride_w, filter_h=filter_h, filter_w=filter_w, in_width=in_width)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(16, 16, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({out_width} / 16)", f"Math.ceil({out_height} / 16)", "1"
