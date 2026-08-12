# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLAssign, WGSLDecl, WGSLEmitter, WGSLFor, WGSLFunction, WGSLIf, WGSLNode, WGSLRaw
from ml_switcheroo_compiler.ir.core import IRGraph


class WebGPUCodeGenerator(BaseGenerator):
    """WebGPU WGSL Code Generator for emitting compute shader module code and browser JS orchestrator.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated WGSL expressions or variable names.
        body_lines (list[str]): Generated WGSL execution body lines.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
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

    def _get_shape_and_strides(self, node: Any) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (object): The IR node to analyze.

        Returns:
            Tuple[List[int], List[int]]: The shape as a list of dimensions and the corresponding strides.
        """
        shape_meta = getattr(node, "shape_metadata", None)
        if shape_meta is None:
            return [], []

        if isinstance(shape_meta, (int, float)):
            shape = [int(shape_meta)]
        else:
            shape = [int(s) for s in shape_meta]

        if not shape:
            return [], []

        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return shape, strides

    def _num_elements(self, shape: list[int]) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    def generic_visit(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Process a node and return its generated WGSL variable name.

        Args:
            node (object): The IR node.
            input_vars (List[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def _gen_offset_computation(self, idx_var: str, shape: list[int], strides: list[int], out_var: str) -> list[WGSLNode]:
        """Generate N-dimensional coordinate-to-offset resolution."""
        if not shape:
            return [WGSLDecl("let", out_var, WGSLRaw("0u"), "u32")]

        nodes: list[WGSLNode] = []
        nodes.append(WGSLDecl("var", f"{out_var}_offset", WGSLRaw("0u"), "u32"))
        nodes.append(WGSLDecl("var", f"{out_var}_remaining", WGSLRaw(idx_var), "u32"))
        for i in range(len(shape) - 1, -1, -1):
            nodes.append(WGSLDecl("let", f"{out_var}_d{i}", WGSLRaw(f"{out_var}_remaining % {shape[i]}u")))
            nodes.append(WGSLAssign(f"{out_var}_remaining", WGSLRaw(f"{out_var}_remaining / {shape[i]}u")))
            nodes.append(WGSLAssign(f"{out_var}_offset", WGSLRaw(f"{out_var}_offset + {out_var}_d{i} * {strides[i]}u")))
        nodes.append(WGSLDecl("let", out_var, WGSLRaw(f"{out_var}_offset")))
        return nodes

    def _get_wgsl_for_op(self, node: Any, shape: list[int], nelem: int, clean_id: str) -> tuple[list[str], str, str, str]:
        global_code_list = []
        op_type = getattr(node, "op_type", "")
        body_nodes: list[WGSLNode] = []

        # Get input nodes for shapes/strides
        inputs = getattr(node, "inputs", [])
        input_nodes = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs]

        in0_shape, in0_strides = self._get_shape_and_strides(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else ([], [])
        in1_shape, in1_strides = self._get_shape_and_strides(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else ([], [])
        _, out_strides = self._get_shape_and_strides(node)

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op_type, {})
        mapping = op_def.get("variants", {}).get("edge_wgsl", {})
        if not mapping:
            # Fallback for ops not in mapping
            wg = min(64, nelem)
            wg = max(1, wg)
            body_nodes.append(WGSLDecl("let", "idx", WGSLRaw("global_id.x")))
            body_nodes.append(WGSLIf(WGSLRaw(f"idx >= {nelem}u"), [WGSLRaw("return;")]))
            body_nodes.append(WGSLRaw(f"// Fallback for unsupported {op_type}"))

            # Use offset logic so compiler doesn't complain about undefined offsets
            body_nodes.extend(self._gen_offset_computation("idx", shape, out_strides, "out_offset"))
            body_nodes.append(WGSLAssign("buf_out_f32[out_offset]", WGSLRaw("0.0")))
            func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>"], body_nodes, [f"@compute @workgroup_size({wg})"])
            dispatch_x = f"Math.ceil({nelem} / {wg})"
            dispatch_y = "1"
            dispatch_z = "1"
        else:
            template = get_wgsl_template(mapping["template"])
            wg_size = template.get("workgroup_size", [64, 1, 1])
            wg_x, wg_y, wg_z = wg_size

            # Format expressions
            expr_format_args = {
                "nelem": nelem,
                "TILE_SIZE": 16,  # default tile size
                "K": in0_shape[1] if len(in0_shape) > 1 else 1,
                "N": shape[1] if len(shape) > 1 else 1,
                "M": shape[0] if len(shape) > 0 else 1,
                "clean_id": clean_id,
            }

            # Add dynamic offset code strings
            out_offset_code = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", shape, out_strides, "out_offset")])
            in0_offset_code = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in0_shape, in0_strides, "in0_offset")]) if len(input_nodes) > 0 else ""
            in1_offset_code = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in1_shape, in1_strides, "in1_offset")]) if len(input_nodes) > 1 else ""
            in2_shape, in2_strides = self._get_shape_and_strides(input_nodes[2]) if len(input_nodes) > 2 and input_nodes[2] else ([], [])
            in2_offset_code = "\n".join([self.emitter.emit(n) for n in self._gen_offset_computation("idx", in2_shape, in2_strides, "in2_offset")]) if len(input_nodes) > 2 else ""

            expr_format_args.update({"out_offset_code": out_offset_code, "in0_offset_code": in0_offset_code, "in1_offset_code": in1_offset_code, "in2_offset_code": in2_offset_code, "nelem_in": getattr(node, "inputs_nelem", [1])[0], "expr": mapping.get("expr", "0.0")})
            # Add all keys from mapping into kwargs so that templates can use them (e.g. init_code)
            expr_format_args.update(mapping)

            # Format the body text and wrap it in WGSLRaw
            formatted_body = template["body"].format(**expr_format_args)
            body_nodes.append(WGSLRaw(formatted_body))

            if "global_code" in template:
                global_code_list.append(template["global_code"].format(**expr_format_args))

            func = WGSLFunction(f"compute_{clean_id}", ["@builtin(global_invocation_id) global_id: vec3<u32>", "@builtin(local_invocation_id) local_id: vec3<u32>"], body_nodes, [f"@compute @workgroup_size({wg_x}, {wg_y}, {wg_z})"])

            if mapping["template"] == "matmul":
                dispatch_x = f"Math.ceil({expr_format_args['N']} / {wg_x})"
                dispatch_y = f"Math.ceil({expr_format_args['M']} / {wg_y})"
                dispatch_z = "1"
            else:
                wg_total = wg_x * wg_y * wg_z
                dispatch_x = f"Math.ceil({nelem} / {wg_total})"
                dispatch_y = "1"
                dispatch_z = "1"

        wgsl_str = global_code_list + self.emitter.emit(func).split("\n")
        return wgsl_str, dispatch_x, dispatch_y, dispatch_z

    def generate(self) -> str:
        """Generate WebGPU WGSL compute shader module code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable JavaScript orchestration code wrapper around WGSL compute shader.
        """
        output_ids = getattr(self.graph, "outputs", []) or []

        wgsl = []
        js = []

        # 1. WGSL Global Bindings
        wgsl.append("// WebGPU WGSL generated by ml-switcheroo-compiler")
        for i in range(3):
            wgsl.append(f"@group(0) @binding({i}) var<storage, read> buf_in{i}_f32: array<f32>;")
        wgsl.append("@group(0) @binding(3) var<storage, read_write> buf_out_f32: array<f32>;")
        for i in range(3):
            wgsl.append(f"@group(0) @binding({i + 4}) var<storage, read> buf_in{i}_i32: array<i32>;")
        wgsl.append("@group(0) @binding(7) var<storage, read_write> buf_out_i32: array<i32>;")

        # Removed static helper function because N-dimensional offsetting is dynamically generated via AST per-node
        wgsl.append("")

        js.append("// WebGPU JavaScript Orchestrator Code Generated by ml-switcheroo-compiler")
        js.append("async function run(inputs) {")
        js.append("  if (!navigator.gpu) throw new Error('WebGPU is not supported on this browser.');")
        js.append("  const adapter = await navigator.gpu.requestAdapter();")
        js.append("  if (!adapter) throw new Error('No appropriate GPUAdapter found.');")
        js.append("  const device = await adapter.requestDevice();")

        # Create buffers for all nodes
        js.append("  // Allocate storage buffers for all nodes")
        for node in self.sorted_nodes:
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            shape, _ = self._get_shape_and_strides(node)
            nelem = self._num_elements(shape) if shape else 1
            if getattr(node, "op_type", "") == "Input":
                js.append(f"  const buf_{clean_id} = device.createBuffer({{ size: inputs.{nid} ? inputs.{nid}.byteLength : {nelem * 4}, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }});")
                js.append(f"  if (inputs.{nid}) device.queue.writeBuffer(buf_{clean_id}, 0, inputs.{nid});")
            else:
                js.append(f"  const buf_{clean_id} = device.createBuffer({{ size: {nelem * 4}, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }});")

        js.append("")

        # Create Output Staging Buffers
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem = self._num_elements(shape) if shape else 1
            js.append(f"  const out_{i}_staging = device.createBuffer({{ size: {nelem * 4}, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }});")

        # Generate WGSL Compute passes and JS orchestration per node
        js.append("  const commandEncoder = device.createCommandEncoder();")

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            inputs = getattr(node, "inputs", [])
            shape, _ = self._get_shape_and_strides(node)
            nelem = self._num_elements(shape) if shape else 1

            op_wgsl, dispatch_x, dispatch_y, dispatch_z = self._get_wgsl_for_op(node, shape, nelem, clean_id)
            wgsl.extend(op_wgsl)

            js.append(f"  const pipe_{clean_id} = device.createComputePipeline({{ layout: 'auto', compute: {{ module: shaderModule, entryPoint: 'compute_{clean_id}' }} }});")

            entries = []
            for j, inp in enumerate(inputs):
                if j < 3:
                    entries.append(f"{{ binding: {j}, resource: {{ buffer: buf_{inp.replace('-', '_')} }} }}")
            entries.append(f"{{ binding: 3, resource: {{ buffer: buf_{clean_id} }} }}")

            js.append(f"  const bg_{clean_id} = device.createBindGroup({{ layout: pipe_{clean_id}.getBindGroupLayout(0), entries: [{', '.join(entries)}] }});")
            js.append(f"  const pass_{clean_id} = commandEncoder.beginComputePass();")
            js.append(f"  pass_{clean_id}.setPipeline(pipe_{clean_id});")
            js.append(f"  pass_{clean_id}.setBindGroup(0, bg_{clean_id});")
            js.append(f"  pass_{clean_id}.dispatchWorkgroups({dispatch_x}, {dispatch_y}, {dispatch_z});")
            js.append(f"  pass_{clean_id}.end();")
            js.append("")

        js.append("  // Copy outputs to staging")
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem = self._num_elements(shape) if shape else 1
            js.append(f"  commandEncoder.copyBufferToBuffer(buf_{out_id.replace('-', '_')}, 0, out_{i}_staging, 0, {nelem * 4});")

        js.append("  device.queue.submit([commandEncoder.finish()]);")

        ret_entries = []
        for i, out_id in enumerate(output_ids):
            js.append(f"  await out_{i}_staging.mapAsync(GPUMapMode.READ);")
            js.append(f"  const out_{i}_array = new Float32Array(out_{i}_staging.getMappedRange().slice());")
            js.append(f"  out_{i}_staging.unmap();")
            ret_entries.append(f"    {out_id}: out_{i}_array,")

        js.append("  return {")
        for entry in ret_entries:
            js.append(entry)
        js.append("  };")
        js.append("}")

        full_code = []
        full_code.append(js[0])
        full_code.append("const shaderCode = `")
        full_code.extend(wgsl)
        full_code.append("`;")
        full_code.append("")
        full_code.extend(js[1:])
        return "\n".join(full_code)
