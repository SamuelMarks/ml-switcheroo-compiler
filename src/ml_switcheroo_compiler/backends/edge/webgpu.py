"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

from __future__ import annotations

from typing import Optional, cast  # noqa: F401

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLEmitter
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("webgpu")
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
            delegates (list[CodeGeneratorVisitor] | None): Visitor delegates.
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
        lowered: str = str(dtype).lower()
        if lowered == "float64":
            import warnings

            warnings.warn(
                "WebGPU WGSL lacks native float64 support; using emulated double-precision vec2<f32> representation.",
                UserWarning,
                stacklevel=2,
            )
            return "vec2<f32>"
        return {
            "float32": "f32",
            "int32": "i32",
            "bool": "bool",
        }.get(lowered, "f32")

    def _get_binding_for_input(self, input_idx: int) -> int:
        """Get the WebGPU binding index for an input tensor.

        Args:
            input_idx (int): Zero-based index of the input.

        Returns:
            int: Binding index in bind group 0.
        """
        if input_idx < 3:
            return input_idx
        return input_idx + 1

    def _get_binding_for_output(self, output_idx: int) -> int:
        """Get the WebGPU binding index for an output tensor.

        Args:
            output_idx (int): Zero-based index of the output.

        Returns:
            int: Binding index in bind group 0.
        """
        if output_idx == 0:
            return 3
        return 5 + output_idx

    def _generate_bind_group_declarations(self, max_inputs: int, max_outputs: int = 1) -> str:
        """Generate dynamic WGSL bind group declarations for storage buffers.

        Args:
            max_inputs (int): Maximum input count needed across all graph nodes.
            max_outputs (int): Maximum output count needed across all graph nodes.

        Returns:
            str: Dynamic WGSL declarations for storage buffer bind group 0.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import generate_dynamic_bindgroup_declarations

        return generate_dynamic_bindgroup_declarations(num_inputs=max_inputs, num_outputs=max_outputs)

    def _get_shape_and_strides(self, node: IRNode | None) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (IRNode | None): The IR node to analyze.

        Returns:
            Tuple[List[int], List[int]]: The shape as a list of dimensions and the corresponding strides.
        """
        shape_meta = getattr(node, "shape_metadata", None)
        if shape_meta is None:
            return [], []

        if isinstance(shape_meta, (int, float)):
            shape: list[int] = [int(shape_meta)]
        else:
            shape = [int(s) for s in shape_meta]

        if not shape:
            return [], []

        # Check if custom memory strides are explicitly specified in attributes
        custom_strides = getattr(node, "attributes", {}).get("strides") if node else None
        if custom_strides is not None and isinstance(custom_strides, (list, tuple)) and len(custom_strides) == len(shape):
            return shape, [int(st) for st in custom_strides]

        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return shape, strides

    def _num_elements(self, shape: list[int]) -> int:
        """Execute num elements for compiler processing.

        Args:
        self (TensorOrScalar): The self parameter.
        shape (list[int]): The shape parameter.

        Returns:
        int: Result.
        """
        n: int = 1
        for s in shape:
            n *= s
        return n

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Process a node and return its generated WGSL variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (List[str]): Names of the input variables.
            **kwargs (EagerValue): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def _gen_offset_computation(self, idx_var: str, shape: list[int], strides: list[int], out_var: str, base_offset_words: int = 0) -> list[str]:
        """Generate N-dimensional coordinate-to-offset resolution.

        Args:
            idx_var: The idx_var parameter.
            shape: The shape parameter.
            strides: The strides parameter.
            out_var: The out_var parameter.
            base_offset_words: The base_offset_words parameter.

        Returns:
            list[str]: Computed result.
        """
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

    def _resolve_wgsl_op_name(self, op_type: str) -> str:
        """Resolve an IR operation type name to a canonical WGSL op name.

        Args:
            op_type (str): Raw operation type from the IR node.

        Returns:
            str: Canonical operation name.
        """
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        if op_type in OPS_REGISTRY:
            return op_type
        prefix_map: dict[str, str] = {
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
        clean_prefix: str = op_type.split("_")[0].lower() if "_" in op_type else op_type.lower()
        return prefix_map.get(clean_prefix, op_type)

    def _resolve_wgsl_mapping(self, node: IRNode, op_type: str, resolved_op: str) -> dict[str, str]:
        """Resolve the WGSL template mapping dictionary for an operation node.

        Args:
            node (IRNode): The IR node being lowered.
            op_type (str): Original IR operation name.
            resolved_op (str): Canonical resolved operation name.

        Returns:
            dict[str, str]: WGSL template mapping configuration.

        Raises:
            CompilationError: If the operation cannot be mapped to a valid WGSL kernel.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_op_mapping
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        decl_mapping: dict[str, str] = cast(dict[str, str], dict(get_wgsl_op_mapping(resolved_op) or get_wgsl_op_mapping(op_type)))
        if decl_mapping:
            mapping: dict[str, str] = decl_mapping
            if resolved_op == "Constant":
                const_val: float = float(getattr(node, "attributes", {}).get("value", 1.0))
                mapping["expr"] = mapping.get("expr", "{const_val}f").replace("{const_val}", str(const_val))
        else:
            op_def = OPS_REGISTRY.get(resolved_op, OPS_REGISTRY.get(op_type, {}))
            mapping = dict(op_def.get("variants", {}).get("edge_wgsl", {}))

        if not mapping:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"Operation '{op_type}' cannot be mapped to a valid WGSL compute kernel.")
        return mapping

    def _emit_node_offset_code(
        self,
        target_node: IRNode | None,
        shape: list[int],
        strides: list[int],
        out_var: str,
    ) -> str:
        """Emit WGSL coordinate-to-offset code for a given tensor node.

        Args:
            target_node (IRNode | None): Node whose buffer offset and shape are used.
            shape (list[int]): Tensor shape dimensions.
            strides (list[int]): Tensor memory strides.
            out_var (str): Output WGSL offset variable name.

        Returns:
            str: Emitted WGSL statements for computing the linear word offset.
        """
        offset_words: int = int(getattr(target_node, "attributes", {}).get("buffer_offset", 0)) // 4 if target_node else 0
        stmts: list[str] = self._gen_offset_computation("idx", shape, strides, out_var, offset_words)
        return "\n".join([self.emitter.emit(n) for n in stmts])

    def _build_offset_format_args(
        self,
        node: IRNode,
        shape: list[int],
        out_strides: list[int],
        input_nodes: list[IRNode | None],
        in0_shape: list[int],
        in0_strides: list[int],
        in1_shape: list[int],
        in1_strides: list[int],
    ) -> dict[str, str]:
        """Build dynamic offset WGSL code snippets for output and input tensors.

        Args:
            node (IRNode): Current operation node.
            shape (list[int]): Output tensor shape.
            out_strides (list[int]): Output tensor strides.
            input_nodes (list[IRNode | None]): Resolved input IR nodes.
            in0_shape (list[int]): First input shape.
            in0_strides (list[int]): First input strides.
            in1_shape (list[int]): Second input shape.
            in1_strides (list[int]): Second input strides.

        Returns:
            dict[str, str]: Mapping of offset code placeholders to WGSL code strings.
        """
        out_code: str = self._emit_node_offset_code(node, shape, out_strides, "out_offset")
        in0_code: str = self._emit_node_offset_code(input_nodes[0], in0_shape, in0_strides, "in0_offset") if len(input_nodes) > 0 else ""
        in1_code: str = self._emit_node_offset_code(input_nodes[1], in1_shape, in1_strides, "in1_offset") if len(input_nodes) > 1 else ""
        in2_shape, in2_strides = self._get_shape_and_strides(input_nodes[2]) if len(input_nodes) > 2 and input_nodes[2] else ([], [])
        in2_code: str = self._emit_node_offset_code(input_nodes[2], in2_shape, in2_strides, "in2_offset") if len(input_nodes) > 2 else ""
        return {
            "out_offset_code": out_code,
            "in0_offset_code": in0_code,
            "in1_offset_code": in1_code,
            "in2_offset_code": in2_code,
        }

    def _populate_tensor_dim_args(
        self,
        expr_format_args: dict[str, EagerValue],
        shape: list[int],
        in0_shape: list[int],
        in1_shape: list[int],
    ) -> None:
        """Populate 4D spatial tensor dimension arguments for convolution and pooling templates.

        Args:
            expr_format_args (dict[str, EagerValue]): Template format arguments dictionary to update.
            shape (list[int]): Output tensor shape.
            in0_shape (list[int]): First input tensor shape.
            in1_shape (list[int]): Second input tensor shape.

        Returns:
            None: Updates expr_format_args in-place.
        """
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

    def _populate_window_and_stride_args(
        self,
        expr_format_args: dict[str, EagerValue],
        attrs: dict[str, EagerValue],
        mapping: dict[str, str],
    ) -> None:
        """Populate window size, stride, and default template format arguments.

        Args:
            expr_format_args (dict[str, EagerValue]): Template format arguments dictionary to update.
            attrs (dict[str, EagerValue]): Node attributes dictionary.
            mapping (dict[str, str]): Operation WGSL template mapping.

        Returns:
            None: Updates expr_format_args in-place.
        """
        if "window_size" in attrs:
            win = attrs["window_size"]
            if isinstance(win, (list, tuple)):
                expr_format_args["window_h"] = win[0]
                expr_format_args["window_w"] = win[1] if len(win) > 1 else win[0]
            else:
                expr_format_args["window_h"] = win
                expr_format_args["window_w"] = win

        stride: EagerValue = attrs.get("stride", 1)
        if isinstance(stride, (list, tuple)):
            expr_format_args["stride_h"] = stride[0]
            expr_format_args["stride_w"] = stride[1] if len(stride) > 1 else stride[0]
        else:
            expr_format_args["stride_h"] = stride
            expr_format_args["stride_w"] = stride

        default_one_keys: tuple[str, ...] = (
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
        )
        for k in default_one_keys:
            expr_format_args.setdefault(k, 1)

        expr_format_args.setdefault("start_idx", 0)
        expr_format_args.setdefault("post_loop_code", "")
        expr_format_args.setdefault("result_var", "acc")
        expr_format_args.setdefault("loop_code", "acc = acc + buf_in0_f32[i];")
        expr_format_args.setdefault("init_code", "var acc: f32 = 0.0f;")

        if mapping.get("template") == "tiled_matmul":
            expr_format_args["TILE_M"] = 16
            expr_format_args["TILE_N"] = 16
            expr_format_args["TILE_K"] = 16

    def _format_dispatch_dims(
        self,
        mapping: dict[str, str],
        expr_format_args: dict[str, EagerValue],
        nelem: int,
        wg_size: list[int],
    ) -> tuple[str, str, str]:
        """Format the (x, y, z) WebGPU workgroup dispatch dimension expressions.

        Args:
            mapping (dict[str, str]): Operation WGSL mapping dictionary.
            expr_format_args (dict[str, EagerValue]): Populated template format arguments.
            nelem (int): Total number of output elements.
            wg_size (list[int]): Workgroup size [x, y, z].

        Returns:
            tuple[str, str, str]: Formatted dispatch_x, dispatch_y, and dispatch_z expressions.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_webgpu_ops

        dispatch_rules = cast("dict[str, dict[str, str]]", get_webgpu_ops().get("dispatch_rules", {}))
        rule: dict[str, str] = dispatch_rules.get(mapping.get("template", "default"), dispatch_rules.get("default", {}))

        fmt_args: dict[str, EagerValue] = dict(expr_format_args)
        fmt_args["nelem"] = nelem
        fmt_args["wg_x"], fmt_args["wg_y"], fmt_args["wg_z"] = wg_size
        dispatch_x: str = rule.get("x", "1").format(**fmt_args)
        dispatch_y: str = rule.get("y", "1").format(**fmt_args)
        dispatch_z: str = rule.get("z", "1").format(**fmt_args)
        return dispatch_x, dispatch_y, dispatch_z

    def _get_wgsl_for_op(self, node: IRNode, shape: list[int], nelem: int, clean_id: str) -> tuple[list[str], str, str, str]:
        """Generate WGSL compute shader lines and dispatch dimensions for an IR node.

        Args:
            node (IRNode): The IR node to lower.
            shape (list[int]): Output tensor shape.
            nelem (int): Number of output elements.
            clean_id (str): Sanitized node identifier.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and (x, y, z) dispatch strings.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        op_type: str = getattr(node, "op_type", "")
        resolved_op: str = self._resolve_wgsl_op_name(op_type)
        if resolved_op in ("Input", "Output"):
            return ([], "1", "1", "1")

        mapping: dict[str, str] = self._resolve_wgsl_mapping(node, op_type, resolved_op)
        inputs: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs]

        in0_shape, in0_strides = self._get_shape_and_strides(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else ([], [])
        in1_shape, in1_strides = self._get_shape_and_strides(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else ([], [])
        _, out_strides = self._get_shape_and_strides(node)

        template = get_wgsl_template(mapping["template"])
        wg_size: list[int] = list(cast("list[int]", template.get("workgroup_size", [64, 1, 1])))
        wg_x, wg_y, wg_z = wg_size

        outer_m: int = self._num_elements(shape[:-1]) if len(shape) > 1 else 1
        inner_n: int = shape[-1] if len(shape) > 0 else 1

        expr_format_args: dict[str, EagerValue] = {
            "nelem": nelem,
            "TILE_SIZE": 16,
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
            "N": inner_n,
            "M": outer_m,
            "clean_id": clean_id,
            "nelem_in": getattr(node, "inputs_nelem", [1])[0],
            "expr": mapping.get("expr", "0.0"),
        }
        expr_format_args.update(self._build_offset_format_args(node, shape, out_strides, input_nodes, in0_shape, in0_strides, in1_shape, in1_strides))
        expr_format_args.update(mapping)

        attrs: dict[str, EagerValue] = getattr(node, "attributes", {})
        expr_format_args.update(attrs)
        self._populate_tensor_dim_args(expr_format_args, shape, in0_shape, in1_shape)
        self._populate_window_and_stride_args(expr_format_args, attrs, mapping)

        formatted_body: str = str(template["body"]).format(**expr_format_args) if template.get("body") else ""
        global_code_list: list[str] = [str(template["global_code"]).format(**expr_format_args)] if template.get("global_code") else []

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"],
            [WGSLRaw(formatted_body)],
            [f"@compute @workgroup_size({wg_x}, {wg_y}, {wg_z})"],
        )
        wgsl_str: list[str] = global_code_list + self.emitter.emit(func).split("\n")
        dispatch_x, dispatch_y, dispatch_z = self._format_dispatch_dims(mapping, expr_format_args, nelem, wg_size)
        return wgsl_str, dispatch_x, dispatch_y, dispatch_z

    def visit_AllReduce(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllReduce.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllReduce", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllGather(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllGather.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllGather", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllToAll(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllToAll.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllToAll", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_ReduceScatter(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WebRTC ReduceScatter.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_Broadcast(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WebRTC Broadcast.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("Broadcast", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_WhileLoop(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Generate WGSL loop constructs via YAML templates with dynamic subgraph lowering.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))  # noqa: F841
        nelem = int(cast("int", kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template = get_wgsl_template("while_loop")
        attrs: dict[str, str] = getattr(node, "attributes", {})

        # Lower body subgraph
        body_graph = attrs.get("body")
        loop_body_lines: list[str] = []
        current_state_var: str = "current_state"

        if body_graph:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(cast("LogicalGraph", body_graph)):
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
        cond_graph = attrs.get("cond")
        condition_expr: str = attrs.get("condition_expr") or (f"current_state {attrs.get('comparator', '<')} {attrs.get('threshold', '10.0')}")
        if cond_graph and "condition_expr" not in attrs:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            comparator: str = "<"
            threshold: str = "10.0"
            for sub_node in topological_sort(cast("LogicalGraph", cond_graph)):
                op_type = getattr(sub_node, "op_type", "")
                if op_type in ("Less", "Lt"):
                    comparator = "<"
                elif op_type in ("Greater", "Gt"):
                    comparator = ">"
                if "threshold" in getattr(sub_node, "attributes", {}):
                    threshold = str(sub_node.attributes["threshold"])
            condition_expr = f"current_state {comparator} {threshold}"

        body: str = str(template["body"]).format(
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

    def visit_Cond(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Generate WGSL if/else block branching via YAML templates with dynamic subgraph lowering.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))  # noqa: F841
        nelem = int(cast("int", kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template = get_wgsl_template("cond")
        attrs: dict[str, str] = getattr(node, "attributes", {})

        def _lower_branch(branch_graph: EagerValue, default_val: str) -> str:
            """Lower a branch subgraph into WGSL strings.

            Args:
                branch_graph (EagerValue): The subgraph to lower.
                default_val (str): The default input value string.

            Returns:
                str: Generated WGSL code string.
            """
            if not branch_graph:
                return f"buf_out_f32[idx] = {default_val};"
            lines: list[str] = []
            current_state: str = "buf_in0_f32[idx]"
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(cast("LogicalGraph", branch_graph)):
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
        body: str = str(template["body"]).format(nelem=nelem, condition_expr=condition_expr, true_body=true_body, false_body=false_body)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_Scan(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Generate WGSL scan via YAML templates.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))  # noqa: F841
        nelem = int(cast("int", kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template, get_wgsl_template  # noqa: F401

        template = get_wgsl_template("scan")

        init_val: str = str(attrs.get("init_val", "0.0")) if (attrs := getattr(node, "attributes", {})) else "0.0"
        scan_op_expr: str = attrs.get("scan_op_expr") or "acc + buf_in0_f32[i]"
        body: str = str(template["body"]).format(nelem=nelem, init_val=init_val, scan_op_expr=scan_op_expr)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(64, 1, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def _compute_max_io_counts(self) -> tuple[int, int]:
        """Compute maximum input and output tensor counts across all graph nodes.

        Args:
            self (WebGPUCodeGenerator): The code generator instance.

        Returns:
            tuple[int, int]: Maximum number of inputs (at least 3) and outputs (at least 1).
        """
        all_inputs_lens: list[int] = [len(getattr(n, "inputs", [])) for n in self.sorted_nodes]
        max_inputs: int = max(all_inputs_lens + [3]) if all_inputs_lens else 3
        max_outputs: int = 1
        for n in self.sorted_nodes:
            raw_outs = getattr(n, "outputs", None) or getattr(n, "attributes", {}).get("outputs")
            if isinstance(raw_outs, (list, tuple)):
                max_outputs = max(max_outputs, len(raw_outs))
        return max_inputs, max_outputs

    def _build_arena_allocations(self) -> tuple[dict[str, str], dict[str, int]]:
        """Map graph nodes to shared memory arenas and compute total byte size per arena.

        Args:
            self (WebGPUCodeGenerator): The code generator instance.

        Returns:
            tuple[dict[str, str], dict[str, int]]: Node-to-arena mapping and arena-to-byte-size mapping.
        """
        node_arenas: dict[str, str] = {}
        for idx, node in enumerate(self.sorted_nodes):
            nid: str = getattr(node, "id", "")
            raw_arena = getattr(node, "attributes", {}).get("buffer_id")
            if raw_arena is not None:
                arena_id: str = str(raw_arena)
            elif getattr(node, "op_type", "") == "Output" and getattr(node, "inputs", []):
                arena_id = node_arenas.get(node.inputs[0], str(idx))
            else:
                arena_id = str(idx)
            node_arenas[nid] = arena_id

        arenas: dict[str, int] = {}
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Output":
                continue
            nid = getattr(node, "id", "")
            arena_id = node_arenas.get(nid, "0")
            offset: int = int(getattr(node, "attributes", {}).get("buffer_offset", 0))
            shape, _ = self._get_shape_and_strides(node)
            size: int = self._num_elements(shape) * 4 if shape else 4
            arenas[arena_id] = max(arenas.get(arena_id, 0), offset + size)
        return node_arenas, arenas

    def _emit_js_arena_and_input_setup(
        self,
        js: list[str],
        node_arenas: dict[str, str],
        arenas: dict[str, int],
        output_ids: list[str],
        webrtc_ops_found: bool,
    ) -> None:
        """Emit JS arena allocations, optional WebRTC init, input writes, and staging buffers.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.
            arenas (dict[str, int]): Mapping from arena ID to required byte size.
            output_ids (list[str]): Graph output node IDs.
            webrtc_ops_found (bool): Whether distributed WebRTC collective ops are present.

        Returns:
            None: Appends initialization statements to js in-place.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        js.append("  // Allocate shared storage arenas")
        for arena_id, total_size in arenas.items():
            js.append(get_js_orchestration_template("allocate_arena").format(arena_id=arena_id, total_size=total_size))

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
                arena_id = node_arenas.get(nid, "0")
                offset: int = int(getattr(node, "attributes", {}).get("buffer_offset", 0))
                js.append(get_js_orchestration_template("write_input").format(nid=nid, arena_id=arena_id, offset=offset))

        js.append("")
        for i, out_id in enumerate(output_ids):
            out_node: IRNode | None = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(get_js_orchestration_template("create_staging").format(i=i, size=nelem * 4))

    def _build_node_bind_entries(
        self,
        node: IRNode,
        inputs: list[str],
        node_arenas: dict[str, str],
    ) -> list[str]:
        """Build JavaScript bind group entry strings for a node's inputs and outputs.

        Args:
            node (IRNode): The operation node.
            inputs (list[str]): Input node IDs for the operation.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.

        Returns:
            list[str]: Formatted JS bind group resource entry literals.
        """
        entries: list[str] = []
        for j, inp in enumerate(inputs):
            inp_arena_id: str = node_arenas.get(inp, "0")
            b_idx: int = self._get_binding_for_input(j)
            entries.append(f"{{ binding: {b_idx}, resource: {{ buffer: buf_arena_{inp_arena_id} }} }}")

        raw_outs = getattr(node, "attributes", {}).get("outputs") or getattr(node, "outputs", None)
        out_nodes: list[str] = [str(o) for o in raw_outs] if isinstance(raw_outs, (list, tuple)) else [getattr(node, "id", "0")]

        for k, out_id in enumerate(out_nodes):
            out_arena_id: str = node_arenas.get(out_id, node_arenas.get(node.id, "0"))
            b_idx = self._get_binding_for_output(k)
            entries.append(f"{{ binding: {b_idx}, resource: {{ buffer: buf_arena_{out_arena_id} }} }}")
        return entries

    def _emit_compute_passes(
        self,
        wgsl: list[str],
        js: list[str],
        node_arenas: dict[str, str],
    ) -> None:
        """Emit WGSL compute shaders and JavaScript compute pass dispatches for all active nodes.

        Args:
            wgsl (list[str]): Accumulator list of WGSL shader lines.
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.

        Returns:
            None: Appends WGSL kernels and JS compute passes in-place.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        js.append("  const commandEncoder = device.createCommandEncoder();")

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")

            inputs: list[str] = getattr(node, "inputs", [])
            shape, _ = self._get_shape_and_strides(node)
            nelem: int = self._num_elements(shape) if shape else 1

            visitor = getattr(self, f"visit_{op_type}", None)
            if visitor is not None:
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = visitor(node, inputs, shape=shape, nelem=nelem, clean_id=clean_id)
            else:
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = self._get_wgsl_for_op(node, shape, nelem, clean_id)
            wgsl.extend(op_wgsl)

            entries: list[str] = self._build_node_bind_entries(node, inputs, node_arenas)
            js.append(
                get_js_orchestration_template("compute_pass").format(
                    clean_id=clean_id,
                    entries=", ".join(entries),
                    dispatch_x=dispatch_x,
                    dispatch_y=dispatch_y,
                    dispatch_z=dispatch_z,
                )
            )
            js.append("")

    def _emit_dynamic_memory_orchestration(self, js: list[str]) -> None:
        """Emit JavaScript dynamic runtime offset calculations and buffer resize logic.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.

        Returns:
            None: Appends dynamic memory orchestration lines to js in-place.
        """
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.config_models import MemorySchemasConfig

        graph_attrs: dict[str, EagerValue] = getattr(self.graph, "attributes", {})
        dynamic_schema = cast("dict[str, EagerValue]", graph_attrs.get("dynamic_memory_schema", {}))
        dynamic_offsets = cast("list[dict[str, str]]", dynamic_schema.get("dynamic_offsets", []))
        if not dynamic_offsets:
            return

        yaml_path: str = os.path.join(os.path.dirname(__file__), "memory_schemas.yaml")
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path) as f:
            schemas: dict[str, EagerValue] = MemorySchemasConfig(**yaml.safe_load(f)).model_dump()

        tpls = cast("dict[str, str]", cast("dict[str, EagerValue]", schemas.get("schemas", {})).get("js_orchestration_templates", {}))
        resize_tpl: str = tpls.get("dynamic_resize", "")
        offset_tpl: str = tpls.get("runtime_offset_calc", "")

        js.append("  // Dynamic Runtime Offsets")
        js.append("  let current_offset = 0;")
        total_computed_size: list[str] = []
        for entry in dynamic_offsets:
            js.append(offset_tpl.format(var_name=entry["var_name"], symbolic_math=entry["symbolic_math"], byte_alignment=4).strip())
            total_computed_size.append(f"({entry['symbolic_math']} * 4)")

        if resize_tpl:
            computed_size_str: str = " + ".join(total_computed_size) if total_computed_size else "0"
            resize_logic: str = "device.createBuffer({ size: new_cap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });"
            growth: float = float(getattr(getattr(getattr(schemas, "get")("schemas", {}), "get")("default", {}), "get")("growth_multiplier", 1.5))
            resize_block: str = resize_tpl.format(computed_size=computed_size_str, arena_id=0, growth_multiplier=growth, resize_logic=resize_logic)
            for line in resize_block.split("\n"):
                js.append("  " + line)

    def _emit_webrtc_and_output_readback(
        self,
        js: list[str],
        node_arenas: dict[str, str],
        output_ids: list[str],
        webrtc_ops_found: bool,
    ) -> None:
        """Emit WebRTC collective operations, staging buffer copies, queue submission, and readback.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.
            output_ids (list[str]): Graph output node IDs.
            webrtc_ops_found (bool): Whether distributed WebRTC collective ops are present.

        Returns:
            None: Appends output readback and return statements to js in-place.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        if webrtc_ops_found:
            collective_ops: tuple[str, ...] = ("AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast")
            for node in self.sorted_nodes:
                op_type: str = getattr(node, "op_type", "")
                if op_type in collective_ops:
                    op_id: str = getattr(node, "id", "")
                    node_inputs: list[str] = getattr(node, "inputs", [])
                    in0: str = node_inputs[0] if node_inputs else "dummy"
                    op_str: str = emit_webrtc_op(op_type, f"buf_arena_{in0}", op_id)
                    if op_str:
                        js.append(f"  // Collectives {op_type}")
                        for line in op_str.split("\n"):
                            js.append("  " + line)

        js.append("  // Copy outputs to staging")
        for i, out_id in enumerate(output_ids):
            out_node: IRNode | None = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            arena_id: str = node_arenas.get(out_id, "0")
            offset: int = int(getattr(out_node, "attributes", {}).get("buffer_offset", 0)) if out_node else 0
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

    def generate(self) -> str:
        r"""Generate WebGPU WGSL compute shader module code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable JavaScript orchestration code wrapper around WGSL compute shader.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        output_ids: list[str] = getattr(self.graph, "outputs", []) or []
        max_inputs, max_outputs = self._compute_max_io_counts()

        wgsl: list[str] = [self._generate_bind_group_declarations(max_inputs, max_outputs), ""]
        js: list[str] = [get_js_orchestration_template("init")]

        node_arenas, arenas = self._build_arena_allocations()
        collective_ops: tuple[str, ...] = ("AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast")
        webrtc_ops_found: bool = any(getattr(n, "op_type", "") in collective_ops for n in self.sorted_nodes)

        self._emit_js_arena_and_input_setup(js, node_arenas, arenas, output_ids, webrtc_ops_found)
        self._emit_compute_passes(wgsl, js, node_arenas)
        self._emit_dynamic_memory_orchestration(js)
        self._emit_webrtc_and_output_readback(js, node_arenas, output_ids, webrtc_ops_found)

        js_str: str = "\n".join(js)
        wgsl_str: str = "\n".join(wgsl)
        return f"const shaderCode = `{wgsl_str}`;\n{js_str}"

    def visit_Linear(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Linear.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_BatchMatMul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for BatchMatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_Attention(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Multi-Head Attention with workgroup shared memory reduction.

        Args:
            node (IRNode): The IR node representing the attention op.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs (shape, nelem, clean_id).

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch (x, y, z) expressions.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        nelem = int(cast("int", kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("attention") or {"body": "buf_out_f32[global_id.x] = 0.0;"}

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

        body: str = str(template["body"]).format(
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

    def visit_MaxPool(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_MaxPool2D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_AvgPool(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_AvgPool2D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (EagerValue): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def _record_node_concrete_shape(
        self,
        node: IRNode,
        concrete: list[EagerValue] | tuple[EagerValue, ...],
        solved_env: dict[str, int],
    ) -> None:
        """Bind symbolic dimensions from a node's existing metadata to concrete telemetry values.

        Args:
            node (IRNode): The IR node being updated.
            concrete (list[EagerValue] | tuple[EagerValue, ...]): Concrete dimension sequence from telemetry.
            solved_env (dict[str, int]): Mapping of symbolic dimension names to resolved integer sizes.

        Returns:
            None: Updates solved_env and node.shape_metadata in-place.
        """
        meta = getattr(node, "shape_metadata", None)
        if meta and isinstance(meta, (list, tuple)) and len(meta) == len(concrete):
            for m_dim, c_dim in zip(meta, concrete):
                if hasattr(m_dim, "node"):
                    solved_env[str(m_dim.node)] = int(cast(int, c_dim))
                elif isinstance(m_dim, str):
                    solved_env[m_dim] = int(cast(int, c_dim))
        node.shape_metadata = tuple(int(x) for x in concrete if isinstance(x, (int, float)))

    def _eval_symbolic_dim(self, dim: EagerValue, solved_env: dict[str, int]) -> int | None:
        """Evaluate a single symbolic or numeric shape dimension using the solved environment.

        Args:
            dim (EagerValue): Symbolic dimension object, variable name, or numeric value.
            solved_env (dict[str, int]): Mapping of symbolic dimension names to resolved integer sizes.

        Returns:
            int | None: Resolved integer dimension size, or None if unresolvable.
        """
        if hasattr(dim, "node") and hasattr(getattr(dim, "node"), "eval"):
            try:
                return int(getattr(dim, "node").eval(solved_env))
            except Exception:
                return int(dim) if isinstance(dim, (int, float)) else 1
        if isinstance(dim, str) and dim in solved_env:
            return solved_env[dim]
        if isinstance(dim, (int, float)):
            return int(dim)
        return None

    def apply_shape_telemetry(self, telemetry: dict[str, EagerValue]) -> None:
        """Update IRGraph node shape metadata and refine symbolic dimensions from client telemetry.

        Args:
            telemetry (dict[str, EagerValue]): Mapping or payload containing learned concrete shapes.

        Returns:
            None: Updates node shape_metadata across the graph in-place.
        """
        shape_data = telemetry.get("shapes", telemetry) if isinstance(telemetry, dict) else {}
        if not isinstance(shape_data, dict):
            return

        solved_env: dict[str, int] = {}
        for node in self.sorted_nodes:
            nid: str = getattr(node, "id", "")
            concrete = shape_data.get(nid)
            if isinstance(concrete, (list, tuple)):
                self._record_node_concrete_shape(node, concrete, solved_env)

        for node in self.sorted_nodes:
            meta = getattr(node, "shape_metadata", None)
            if meta and isinstance(meta, (list, tuple)):
                refined: list[int] = [val for dim in meta if (val := self._eval_symbolic_dim(dim, solved_env)) is not None]
                if refined:
                    node.shape_metadata = tuple(refined)

    def visit_LayerNorm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for LayerNorm.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_Trig(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Trig (Sin/Cos).

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_ReduceSum(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for ReduceSum.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return cast(tuple[list[str], str, str, str], getattr(self, "_get_wgsl_for_op")(node, kwargs.get("shape", []), kwargs.get("nelem", 1), kwargs.get("clean_id", "")))

    def visit_Conv2D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Conv2D.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        nelem = int(cast("int", kwargs.get("nelem", 1)))  # noqa: F841
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template

        template = get_wgsl_template("conv2d")

        in0_shape, in0_strides = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))
        in1_shape, in1_strides = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[1]), None))

        out_width: int = shape[3] if len(shape) == 4 else 1
        out_height: int = shape[2] if len(shape) == 4 else 1
        out_channels: int = shape[1] if len(shape) == 4 else 1
        in_channels: int = in0_shape[1] if len(in0_shape) == 4 else 1
        in_width: int = in0_shape[3] if len(in0_shape) == 4 else 1

        raw_strides = node.attributes.get("strides", (1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1)
        stride_h: int = int(stride_val[0])
        stride_w: int = int(stride_val[1])
        filter_h: int = in1_shape[2] if len(in1_shape) >= 4 else 1
        filter_w: int = in1_shape[3] if len(in1_shape) >= 4 else 1

        body: str = str(template["body"]).format(out_width=out_width, out_height=out_height, in_channels=in_channels, out_channels=out_channels, stride_h=stride_h, stride_w=stride_w, filter_h=filter_h, filter_w=filter_w, in_width=in_width)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(16, 16, 1)"])
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({out_width} / 16)", f"Math.ceil({out_height} / 16)", "1"

    def visit_DepthwiseConv2D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for DepthwiseConv2D.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("DepthwiseConv2D")
        in0_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))
        in1_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[1]), None))

        out_width: int = shape[3] if len(shape) == 4 else 1
        out_height: int = shape[2] if len(shape) == 4 else 1
        channels: int = shape[1] if len(shape) == 4 else 1
        in_width: int = in0_shape[3] if len(in0_shape) == 4 else 1

        raw_strides = getattr(node, "attributes", {}).get("strides", (1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1)
        stride_h: int = int(stride_val[0])
        stride_w: int = int(stride_val[1])
        filter_h: int = in1_shape[2] if len(in1_shape) >= 4 else 1
        filter_w: int = in1_shape[3] if len(in1_shape) >= 4 else 1

        body: str = str(template["body"]).format(
            out_width=out_width,
            out_height=out_height,
            channels=channels,
            stride_h=stride_h,
            stride_w=stride_w,
            filter_h=filter_h,
            filter_w=filter_w,
            in_width=in_width,
        )

        func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(16, 16, 1)"])
        return self.emitter.emit(func).split("\n"), f"Math.ceil({out_width} / 16)", f"Math.ceil({out_height} / 16)", "1"

    def visit_Conv3D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Conv3D.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("Conv3D")
        in0_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))
        in1_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[1]), None))

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        out_channels = shape[1] if len(shape) == 5 else 1
        in_channels = in0_shape[1] if len(in0_shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = getattr(node, "attributes", {}).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(stride_val[0])
        stride_h = int(stride_val[1])
        stride_w = int(stride_val[2])
        filter_d = in1_shape[2] if len(in1_shape) >= 5 else 1
        filter_h = in1_shape[3] if len(in1_shape) >= 5 else 1
        filter_w = in1_shape[4] if len(in1_shape) >= 5 else 1

        body = str(template["body"]).format(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            in_channels=in_channels,
            out_channels=out_channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            filter_d=filter_d,
            filter_h=filter_h,
            filter_w=filter_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(8, 8, 4)"])
        return self.emitter.emit(func).split("\n"), f"Math.ceil({out_w} / 8)", f"Math.ceil({out_h} / 8)", f"Math.ceil({out_d} / 4)"

    def visit_MaxPool3D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool3D.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("MaxPool3D")
        in0_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        channels = shape[1] if len(shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = getattr(node, "attributes", {}).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(stride_val[0])
        stride_h = int(stride_val[1])
        stride_w = int(stride_val[2])

        raw_win = getattr(node, "attributes", {}).get("window", (2, 2, 2))
        win_val = raw_win if isinstance(raw_win, (list, tuple)) else (2, 2, 2)
        window_d = int(win_val[0])
        window_h = int(win_val[1])
        window_w = int(win_val[2])

        body = str(template["body"]).format(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            channels=channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            window_d=window_d,
            window_h=window_h,
            window_w=window_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(8, 8, 4)"])
        return self.emitter.emit(func).split("\n"), f"Math.ceil({out_w} / 8)", f"Math.ceil({out_h} / 8)", f"Math.ceil({out_d} / 4)"

    def visit_AvgPool3D(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool3D.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = cast("list[int]", kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("AvgPool3D")
        in0_shape, _ = self._get_shape_and_strides(next((n for n in self.sorted_nodes if getattr(n, "id", None) == input_vars[0]), None))

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        channels = shape[1] if len(shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = getattr(node, "attributes", {}).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(stride_val[0])
        stride_h = int(stride_val[1])
        stride_w = int(stride_val[2])

        raw_win = getattr(node, "attributes", {}).get("window", (2, 2, 2))
        win_val = raw_win if isinstance(raw_win, (list, tuple)) else (2, 2, 2)
        window_d = int(win_val[0])
        window_h = int(win_val[1])
        window_w = int(win_val[2])

        body = getattr(template["body"], "format")(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            channels=channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            window_d=window_d,
            window_h=window_h,
            window_w=window_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], [WGSLRaw(body)], ["@compute @workgroup_size(8, 8, 4)"])
        return self.emitter.emit(func).split("\n"), f"Math.ceil({out_w} / 8)", f"Math.ceil({out_h} / 8)", f"Math.ceil({out_d} / 4)"

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> dict[str, EagerValue]:
        """Compile IRGraph into ready-to-dispatch WGSL compute shader bundles and pipeline layouts.

        Args:
            graph (IRGraph): The computation graph to compile.
            **kwargs (EagerValue): Compilation options.

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
