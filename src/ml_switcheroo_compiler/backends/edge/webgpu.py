"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
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

    def _get_shape_and_strides(self, node: IRNode | None) -> tuple[list[int], list[int]]:
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

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
            get_js_orchestration_template,
            get_wgsl_op_mapping,
            get_wgsl_template,
        )
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        prefix_map = {
            "add": "Add",
            "sub": "Sub",
            "mul": "Mul",
            "div": "Div",
            "m": "MatMul",
            "matmul": "MatMul",
            "max": "MaxPool2D",
            "maxpool": "MaxPool2D",
            "maxpool2d": "MaxPool2D",
            "avg": "AvgPool2D",
            "avgpool": "AvgPool2D",
            "avgpool2d": "AvgPool2D",
            "conv": "Conv2D",
            "conv2d": "Conv2D",
            "in": "Input",
            "input": "Input",
            "out": "Output",
            "output": "Output",
            "linear": "Linear",
        }
        resolved_op = op_type
        if op_type not in OPS_REGISTRY:
            clean_prefix = op_type.split("_")[0].lower() if "_" in op_type else op_type.lower()
            if clean_prefix in prefix_map:
                resolved_op = prefix_map[clean_prefix]

        if resolved_op in ("Input", "Output"):
            return ([], "1", "1", "1")

        decl_mapping: dict[str, str] = dict(get_wgsl_op_mapping(resolved_op) or get_wgsl_op_mapping(op_type))
        if decl_mapping:
            mapping: dict[str, str] = decl_mapping
            if resolved_op == "Constant":
                const_val: float = float(getattr(node, "attributes", {}).get("value", 1.0))
                mapping["expr"] = mapping.get("expr", "{const_val}f").replace("{const_val}", str(const_val))
        else:
            op_def: dict[str, str] = OPS_REGISTRY.get(resolved_op, OPS_REGISTRY.get(op_type, {}))
            mapping = dict(op_def.get("variants", {}).get("edge_wgsl", {}))

        if not mapping:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"Operation '{op_type}' cannot be mapped to a valid WGSL compute kernel.")

        template: dict[str, str] = get_wgsl_template(mapping["template"])
        wg_size: list[int] = template.get("workgroup_size", [64, 1, 1])
        wg_x, wg_y, wg_z = wg_size

        # Compute dynamic outer batch size M and inner reduction dimension N
        outer_m: int = 1
        if len(shape) > 1:
            for dim in shape[:-1]:
                outer_m *= dim
        inner_n: int = shape[-1] if len(shape) > 0 else 1

        # Format expressions
        expr_format_args: dict[str, int | str] = {
            "nelem": nelem,
            "TILE_SIZE": 16,  # default tile size
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
            "N": inner_n,
            "M": outer_m,
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

        for k in (
            "out_height",
            "out_width",
            "out_channels",
            "channels",
            "in_height",
            "in_width",
            "in_channels",
            "filter_h",
            "filter_w",
            "window_h",
            "window_w",
            "stride_h",
            "stride_w",
        ):
            expr_format_args.setdefault(k, 1)

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

    def visit_Broadcast(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WebRTC Broadcast."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("Broadcast", "buf_in0_f32", op_id)
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
                    current_state_var = temp_var
            if loop_body_lines:
                loop_body_lines.append(f"current_state = {current_state_var};")

        loop_body: str = "\n    ".join(loop_body_lines) if loop_body_lines else "current_state = current_state + buf_in1_f32[idx];"

        # Lower cond subgraph
        cond_graph: str = attrs.get("cond")
        condition_expr: str = attrs.get("condition_expr") or (f"current_state {attrs.get('comparator', '<')} {attrs.get('threshold', '10.0')}")
        if cond_graph and "condition_expr" not in attrs:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            comparator: str = "<"
            threshold: str = "10.0"
            for sub_node in topological_sort(cond_graph):
                op_type = getattr(sub_node, "op_type", "")
                if op_type in ("Less", "Lt"):
                    comparator = "<"
                elif op_type in ("Greater", "Gt"):
                    comparator = ">"
                if "threshold" in getattr(sub_node, "attributes", {}):
                    threshold = str(sub_node.attributes["threshold"])
            condition_expr = f"current_state {comparator} {threshold}"

        body: str = template["body"].format(
            nelem=nelem,
            init_state="buf_in0_f32[idx]",
            condition_expr=condition_expr,
            max_iters=attrs.get("max_iters", 10),
            loop_body=loop_body,
        )

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
                    current_state = temp_var
            lines.append(f"buf_out_f32[idx] = {current_state};")
            return "\n    ".join(lines)

        true_body: str = _lower_branch(attrs.get("then_branch"), "buf_in1_f32[idx]")
        false_body: str = _lower_branch(attrs.get("else_branch"), "buf_in2_f32[idx]")

        condition_expr: str = attrs.get("condition_expr") or (f"buf_in0_f32[idx] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}")
        body: str = template["body"].format(nelem=nelem, condition_expr=condition_expr, true_body=true_body, false_body=false_body)

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

        init_val: str = str(attrs.get("init_val", "0.0")) if (attrs := getattr(node, "attributes", {})) else "0.0"
        scan_op_expr: str = attrs.get("scan_op_expr") or "acc + buf_in0_f32[i]"
        body: str = template["body"].format(nelem=nelem, init_val=init_val, scan_op_expr=scan_op_expr)

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

        node_arenas: dict[str, str] = {}
        for idx, node in enumerate(self.sorted_nodes):
            nid = getattr(node, "id", "")
            raw_arena = getattr(node, "attributes", {}).get("buffer_id")
            if raw_arena is not None:
                arena_id = str(raw_arena)
            elif getattr(node, "op_type", "") == "Output" and getattr(node, "inputs", []):
                arena_id = node_arenas.get(node.inputs[0], str(idx))
            else:
                arena_id = str(idx)
            node_arenas[nid] = arena_id

        # Group nodes by arena
        arenas: dict[str, int] = {}
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Output":
                continue
            nid = getattr(node, "id", "")
            arena_id = node_arenas.get(nid, "0")
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

        webrtc_ops_found: bool = any(getattr(n, "op_type", "") in ["AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast"] for n in self.sorted_nodes)
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
                arena_id: str = node_arenas.get(nid, "0")
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

        inputs_set = set(getattr(self.graph, "inputs", []))

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")
            if op_type not in ("Input", "Output") and nid != "Input" and nid not in inputs_set and clean_id not in inputs_set:
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
                        inp_arena_id: str = node_arenas.get(inp, "0")
                        entries.append(f"{{ binding: {j}, resource: {{ buffer: buf_arena_{inp_arena_id} }} }}")
                out_arena_id: str = node_arenas.get(node.id, "0")
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
                if op_type in ["AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast"]:
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
            arena_id: str = node_arenas.get(out_id, "0")
            offset: int = getattr(out_node, "attributes", {}).get("buffer_offset", 0) if out_node else 0
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(get_js_orchestration_template("copy_output").format(arena_id=arena_id, offset=offset, i=i, size=nelem * 4))

        js.append("  device.queue.submit([commandEncoder.finish()]);")

        ret_entries: list[str] = []
        for i, out_id in enumerate(output_ids):
            js.append(get_js_orchestration_template("read_output").format(i=i))
            ret_entries.append(f'    "{out_id}": out_{i}_array,')

        js.append(get_js_orchestration_template("return_dict").format(returns="\n".join(ret_entries)))
        js.append("}")

        js_str: str = "\n".join(js)
        wgsl_str: list[str] = "\n".join(wgsl)
        return f"const shaderCode = `{wgsl_str}`;\n{js_str}"

    def visit_Linear(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Linear.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_BatchMatMul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for BatchMatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_Attention(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Multi-Head Attention with workgroup shared memory reduction.

        Args:
            node (IRNode): The IR node representing the attention op.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs (shape, nelem, clean_id).

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch (x, y, z) expressions.
        """
        shape: list[int] = kwargs.get("shape", [])
        nelem: int = kwargs.get("nelem", 1)
        clean_id: str = kwargs.get("clean_id", "")

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template: dict[str, str] = get_wgsl_template("attention") or {"body": "buf_out_f32[global_id.x] = 0.0;"}

        head_dim: int = shape[-1] if len(shape) >= 2 else (shape[0] if shape else 1)
        seq_len: int = shape[-2] if len(shape) >= 2 else (nelem // head_dim if head_dim > 0 else 1)
        if seq_len < 1:
            seq_len = 1
        if head_dim < 1:
            head_dim = 1
        scale: float = 1.0 / (head_dim**0.5)

        q_buf: str = "buf_in0_f32"
        k_buf: str = "buf_in1_f32" if len(input_vars) > 1 else "buf_in0_f32"
        v_buf: str = "buf_in2_f32" if len(input_vars) > 2 else "buf_in0_f32"

        body: str = template["body"].format(
            nelem=nelem,
            seq_len=seq_len,
            head_dim=head_dim,
            scale=scale,
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
        )

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(64, 1, 1)"],
        )
        return self.emitter.emit(func).split("\n"), f"Math.ceil({seq_len} / 64)", "1", "1"

    def visit_MaxPool(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_MaxPool2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_AvgPool(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def visit_AvgPool2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", ""))

    def apply_shape_telemetry(self, telemetry: dict[str, object]) -> None:
        """Update IRGraph node shape metadata and refine symbolic dimensions from client telemetry.

        Args:
            telemetry (dict[str, object]): Mapping or payload containing learned concrete shapes.
        """
        shape_data = telemetry.get("shapes", telemetry) if isinstance(telemetry, dict) else {}
        if not isinstance(shape_data, dict):
            return

        solved_env: dict[str, int] = {}
        for node in self.sorted_nodes:
            nid = getattr(node, "id", "")
            if nid in shape_data:
                concrete = shape_data[nid]
                if isinstance(concrete, (list, tuple)):
                    meta = getattr(node, "shape_metadata", None)
                    if meta and isinstance(meta, (list, tuple)) and len(meta) == len(concrete):
                        for m_dim, c_dim in zip(meta, concrete):
                            if hasattr(m_dim, "node"):
                                solved_env[str(m_dim.node)] = int(c_dim)
                            elif isinstance(m_dim, str):
                                solved_env[m_dim] = int(c_dim)
                    node.shape_metadata = tuple(int(x) for x in concrete if isinstance(x, (int, float)))

        # Refine symbolic dimensions across all nodes
        for node in self.sorted_nodes:
            meta = getattr(node, "shape_metadata", None)
            if meta and isinstance(meta, (list, tuple)):
                refined: list[int] = []
                for dim in meta:
                    if hasattr(dim, "node") and hasattr(dim.node, "eval"):
                        try:
                            refined.append(int(dim.node.eval(solved_env)))
                        except Exception:
                            refined.append(int(dim) if isinstance(dim, (int, float)) else 1)
                    elif isinstance(dim, str) and dim in solved_env:
                        refined.append(solved_env[dim])
                    elif isinstance(dim, (int, float)):
                        refined.append(int(dim))
                if refined:
                    node.shape_metadata = tuple(refined)

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

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> dict[str, object]:
        """Compile IRGraph into ready-to-dispatch WGSL compute shader bundles and pipeline layouts.

        Args:
            graph (IRGraph): The computation graph to compile.
            **kwargs (object): Compilation options.

        Returns:
            dict[str, object]: Bundle containing shader code, pipeline layout, and status.
        """
        if graph != self.graph:
            self.graph = graph
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            self.sorted_nodes = topological_sort(graph)

        code_bundle = self.generate()
        return {
            "format": "webgpu_wgsl_bundle",
            "code": code_bundle,
            "pipeline_layout": dict(getattr(self, "var_map", {})),
            "status": "ready_to_dispatch",
        }

    def generate_training_step(
        self,
        wrt_inputs: list[str] | None = None,
        target_output: str | None = None,
    ) -> str:
        """Compile client-side coupled forward-backward WGSL passes for in-browser training.

        Args:
            wrt_inputs (Optional[list[str]]): Input variable IDs with respect to which gradients are calculated.
            target_output (Optional[str]): Target output scalar or loss node ID to differentiate.

        Returns:
            str: Coupled forward-backward JavaScript orchestrator module source code.

        Raises:
            ValueError: If target output cannot be identified in the graph.
        """
        from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

        fwd_code: str = self.generate()

        outputs: list[str] = getattr(self.graph, "outputs", []) or []
        if target_output is not None:
            loss_id: str = target_output
        elif outputs:
            loss_id = outputs[0]
        else:
            raise ValueError("Target output cannot be identified from empty graph outputs.")

        wrt: list[str] = wrt_inputs if wrt_inputs is not None else list(getattr(self.graph, "inputs", []))
        bwd_graph: IRGraph = graph_grad(self.graph, wrt, loss_id)
        bwd_graph.inputs = [nid for nid, n in getattr(bwd_graph, "nodes", {}).items() if getattr(n, "op_type", "") == "Input"]

        bwd_generator: WebGPUCodeGenerator = WebGPUCodeGenerator(bwd_graph)
        bwd_code: str = bwd_generator.generate()

        return (
            "// Coupled forward-backward WebGPU execution module generated by ml-switcheroo-compiler\n"
            f"{fwd_code}\n"
            "// Backward pass:\n"
            f"const bwd_execute = (() => {{\n{bwd_code}\n  return execute;\n}})();\n\n"
            "async function execute_train_step(device, inputs, params, learning_rate = 0.01) {\n"
            "  const fwd_results = await execute(device, inputs);\n"
            "  const bwd_inputs = { ...inputs, ...fwd_results };\n"
            "  const grads = await bwd_execute(device, bwd_inputs);\n"
            "  for (const [pKey, pBuf] of Object.entries(params || {})) {\n"
            "    if (grads[pKey]) {\n"
            "      for (let i = 0; i < pBuf.length; i++) {\n"
            "        pBuf[i] -= learning_rate * grads[pKey][i];\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "  return { outputs: fwd_results, gradients: grads };\n"
            "}\n"
        )
