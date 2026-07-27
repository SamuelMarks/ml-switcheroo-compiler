# ruff: noqa: E501, C901, PLR0912, PLR0915
"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

import uuid
from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
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

    def _get_shape_and_strides(self, node: object) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (object): The IR node to analyze.

        Returns:
            tuple[list[int], list[int]]: The shape as a list of dimensions and the corresponding strides.
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

    def _generate_coord_helpers(self) -> str:
        """Generate WGSL helper functions for N-dimensional coordinate-to-offset calculation.

        Returns:
            str: WebGPU WGSL helper function definitions.
        """
        helpers = []
        for node in self.sorted_nodes:
            nid = getattr(node, "id", "")
            if not nid:
                continue

            shape, strides = self._get_shape_and_strides(node)
            if len(shape) <= 1:
                continue

            func_name = f"get_offset_{nid.replace('-', '_')}"
            lines = [f"fn {func_name}(idx: u32) -> u32 {{"]

            terms = []
            for i, (dim, stride) in enumerate(zip(shape, strides)):
                s_i = strides[i]
                div_str = f" / {s_i}u" if s_i > 1 else ""
                lines.append(f"  let c_{i} = (idx{div_str}) % {dim}u;")
                terms.append(f"c_{i} * {stride}u")

            offset_expr = " + ".join(terms)
            lines.append(f"  return {offset_expr};")
            lines.append("}")
            helpers.append("\n".join(lines))

        return "\n\n".join(helpers)

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated WGSL variable name.

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

        if op_type == "Input":
            arg_idx = len(self.var_map)
            shape, _ = self._get_shape_and_strides(node)
            if len(shape) > 1:
                arg_name = f"in_{arg_idx}[get_offset_{nid.replace('-', '_')}(idx)]"
            else:
                arg_name = f"in_{arg_idx}[idx]"
            self.var_map[nid] = arg_name
            return arg_name

        if op_type == "Constant":
            val = node.attributes.get("value", 0.0)
            res_var = f"v_{nid.replace('-', '_')}"
            self.var_map[nid] = res_var
            self.body_lines.append(f"  let {res_var} = {val};")
            return res_var

        # Map binary and unary math operations
        op_map = {
            "Add": "+",
            "Subtract": "-",
            "Multiply": "*",
            "TrueDivide": "/",
            "Div": "/",
        }

        res_var = f"v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var

        in_vars_mapped = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]

        if op_type in op_map:
            operator = op_map[op_type]
            expr = f" {operator} ".join(in_vars_mapped)
            self.body_lines.append(f"  let {res_var} = {expr};")
        elif op_type in ("Exp", "Log"):
            func_name = op_type.lower()
            self.body_lines.append(f"  let {res_var} = {func_name}({in_vars_mapped[0]});")
        elif op_type in ("Negative", "Neg"):
            self.body_lines.append(f"  let {res_var} = -{in_vars_mapped[0]};")
        else:
            # Dynamic fallback
            args_str = ", ".join(in_vars_mapped)
            self.body_lines.append(f"  let {res_var} = {op_type.lower()}({args_str});")

        return res_var

    def _generate_js_orchestrator(self, wgsl_code_str: str, input_nodes: list[object], output_ids: list[str], out_idx: int, total_size: int) -> str:
        """Generate the JS orchestration wrapper around WGSL code."""
        # Build JavaScript Orchestrator boilerplate
        js_code = []
        js_code.append("// WebGPU JavaScript Orchestrator Code Generated by ml-switcheroo-compiler")
        js_code.append("const shaderCode = `")
        js_code.append(wgsl_code_str)
        js_code.append("`;")
        js_code.append("")
        js_code.append("async function run(inputs) {")
        js_code.append("  if (!navigator.gpu) {")
        js_code.append("    throw new Error('WebGPU is not supported on this browser.');")
        js_code.append("  }")
        js_code.append("  const adapter = await navigator.gpu.requestAdapter();")
        js_code.append("  if (!adapter) {")
        js_code.append("    throw new Error('No appropriate GPUAdapter found.');")
        js_code.append("  }")
        js_code.append("  const device = await adapter.requestDevice();")
        js_code.append("")
        js_code.append("  const shaderModule = device.createShaderModule({ code: shaderCode });")
        js_code.append("")

        # Create input buffers in JS
        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", f"n{idx}")
            js_code.append(f"  const in_{idx}_buffer = device.createBuffer({{")
            js_code.append(f"    size: inputs.{nid}.byteLength,")
            js_code.append("    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,")
            js_code.append("  });")
            js_code.append(f"  device.queue.writeBuffer(in_{idx}_buffer, 0, inputs.{nid});")
            js_code.append("")

        # Create output buffers and staging buffers in JS
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([total_size], [])
            num_elements = 1
            for d in shape:
                num_elements *= d
            byte_length = num_elements * 4  # assuming float32 (4 bytes)
            js_code.append(f"  const out_{i}_buffer = device.createBuffer({{")
            js_code.append(f"    size: {byte_length},")
            js_code.append("    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,")
            js_code.append("  });")
            js_code.append(f"  const out_{i}_staging = device.createBuffer({{")
            js_code.append(f"    size: {byte_length},")
            js_code.append("    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,")
            js_code.append("  });")
            js_code.append("")

        # Setup compute pipeline
        js_code.append("  const pipeline = device.createComputePipeline({")
        js_code.append("    layout: 'auto',")
        js_code.append("    compute: { module: shaderModule, entryPoint: 'main' },")
        js_code.append("  });")
        js_code.append("")

        # Create bind group
        js_code.append("  const bindGroup = device.createBindGroup({")
        js_code.append("    layout: pipeline.getBindGroupLayout(0),")
        js_code.append("    entries: [")
        for idx in range(len(input_nodes)):
            js_code.append(f"      {{ binding: {idx}, resource: {{ buffer: in_{idx}_buffer }} }},")
        for i in range(len(output_ids)):
            js_code.append(f"      {{ binding: {out_idx + i}, resource: {{ buffer: out_{i}_buffer }} }},")
        js_code.append("    ],")
        js_code.append("  });")
        js_code.append("")

        # Dispatch pass
        js_code.append("  const commandEncoder = device.createCommandEncoder();")
        js_code.append("  const passEncoder = commandEncoder.beginComputePass();")
        js_code.append("  passEncoder.setPipeline(pipeline);")
        js_code.append("  passEncoder.setBindGroup(0, bindGroup);")
        js_code.append(f"  passEncoder.dispatchWorkgroups(Math.ceil({total_size} / 64));")
        js_code.append("  passEncoder.end();")
        js_code.append("")

        # Copy buffer to buffer for outputs
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([total_size], [])
            num_elements = 1
            for d in shape:
                num_elements *= d
            byte_length = num_elements * 4
            js_code.append(f"  commandEncoder.copyBufferToBuffer(out_{i}_buffer, 0, out_{i}_staging, 0, {byte_length});")

        js_code.append("  device.queue.submit([commandEncoder.finish()]);")
        js_code.append("")

        # Map staging buffers and read back
        ret_obj_entries = []
        for i, out_id in enumerate(output_ids):
            js_code.append(f"  await out_{i}_staging.mapAsync(GPUMapMode.READ);")
            js_code.append(f"  const out_{i}_array = new Float32Array(out_{i}_staging.getMappedRange().slice());")
            js_code.append(f"  out_{i}_staging.unmap();")
            ret_obj_entries.append(f"    {out_id}: out_{i}_array,")

        js_code.append("")
        js_code.append("  return {")
        for entry in ret_obj_entries:
            js_code.append(entry)
        js_code.append("  };")
        js_code.append("}")

        return "\n".join(js_code)

    def generate(self) -> str:
        """Generate WebGPU WGSL compute shader module code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable JavaScript orchestration code wrapper around WGSL compute shader.
        """
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids = getattr(self.graph, "outputs", []) or []

        self.code = []
        self.body_lines = []

        # Generate WGSL shader structure first
        wgsl_lines = []

        # Declare input binding parameters
        for idx, node in enumerate(input_nodes):
            meta_dtype = self._map_type(getattr(node, "dtype", "float32"))
            wgsl_lines.append(f"@group(0) @binding({idx}) var<storage, read> in_{idx}: array<{meta_dtype}>;")
            nid = getattr(node, "id", "")
            shape, _ = self._get_shape_and_strides(node)
            if len(shape) > 1:
                self.var_map[nid] = f"in_{idx}[get_offset_{nid.replace('-', '_')}(idx)]"
            else:
                self.var_map[nid] = f"in_{idx}[idx]"

        # Declare output binding parameters
        out_idx = len(input_nodes)
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            meta_dtype = self._map_type(getattr(out_node, "dtype", "float32")) if out_node else "f32"
            wgsl_lines.append(f"@group(0) @binding({out_idx + i}) var<storage, read_write> out_{i}: array<{meta_dtype}>;")

        # Append Coordinate Helpers to WGSL
        coord_helpers = self._generate_coord_helpers()
        if coord_helpers:
            wgsl_lines.append("")
            wgsl_lines.append(coord_helpers)

        wgsl_lines.append("")
        wgsl_lines.append("@compute @workgroup_size(64)")
        wgsl_lines.append("fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {")
        wgsl_lines.append("  let idx = global_id.x;")

        # Emit intermediate operations
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        # Add all body lines to WGSL
        for line in self.body_lines:
            wgsl_lines.append(line)

        # Emit output assignments in WGSL
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            res_var = self.var_map.get(out_id, out_id)
            if len(shape) > 1:
                wgsl_lines.append(f"  out_{i}[get_offset_{out_id.replace('-', '_')}(idx)] = {res_var};")
            else:
                wgsl_lines.append(f"  out_{i}[idx] = {res_var};")

        wgsl_lines.append("}")

        wgsl_code_str = "\n".join(wgsl_lines)

        # Determine total output array size for orchestration
        total_size = 1
        if output_ids:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == output_ids[0]), None)
            if out_node:
                shape, _ = self._get_shape_and_strides(out_node)
                for dim in shape:
                    total_size *= dim

        return self._generate_js_orchestrator(wgsl_code_str, input_nodes, output_ids, out_idx, total_size)
