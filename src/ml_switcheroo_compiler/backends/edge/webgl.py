# ruff: noqa: E501, C901, PLR0912, PLR0915
"""WebGL GLSL Target Emission with Multi-Dimensional Texture-Grid Transformations and JS Orchestration."""

import uuid
from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class WebGLCodeGenerator(BaseGenerator):
    """WebGL GLSL Code Generator for emitting fragment shader module code and browser WebGL2 JS orchestrator.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated GLSL expressions or variable names.
        body_lines (list[str]): Generated GLSL execution body lines.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
        """Initialize WebGLCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines: list[str] = []

    def _get_shape_and_strides(self, node: object) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (object): The IR node to analyze.

        Returns:
            tuple[list[int], list[int]]: Shape and strides.
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

    def _generate_texture_helpers(self, input_nodes: list[Any]) -> str:
        """Generate GLSL coordinate lookup helpers for multi-dimensional textures.

        Args:
            input_nodes (list[Any]): List of input nodes.

        Returns:
            str: WebGL GLSL function definitions for coordinate transformations.
        """
        helpers = []
        for idx, node in enumerate(input_nodes):
            shape, strides = self._get_shape_and_strides(node)
            if len(shape) <= 1:
                continue

            # Compute dimension sizes and strides
            func_name = f"get_val_in_{idx}"
            lines = [f"float {func_name}(int idx) {{"]

            # Multi-dimensional coordinate index decoding
            terms = []
            for i, (dim, stride) in enumerate(zip(shape, strides)):
                s_i = strides[i]
                div_str = f" / {s_i}" if s_i > 1 else ""
                lines.append(f"  int c_{i} = (idx{div_str}) % {dim};")
                terms.append(f"c_{i} * {stride}")

            offset_expr = " + ".join(terms)
            lines.append(f"  int offset = {offset_expr};")

            # Map flat offset to 2D texture width/height (assuming square or sqrt packing)
            lines.append(f"  ivec2 size = textureSize(in_{idx}, 0);")
            lines.append("  int x = offset % size.x;")
            lines.append("  int y = offset / size.x;")
            lines.append("  vec2 target_uv = (vec2(x, y) + vec2(0.5)) / vec2(size);")
            lines.append(f"  return texture(in_{idx}, target_uv).r;")
            lines.append("}")
            helpers.append("\n".join(lines))

        return "\n\n".join(helpers)

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated GLSL variable name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return "glsl_op"

        op_type = getattr(node, "op_type", "")
        nid = getattr(node, "id", str(uuid.uuid4()))

        if op_type == "Input":
            arg_idx = len(self.var_map)
            shape, _ = self._get_shape_and_strides(node)
            if len(shape) > 1:
                arg_name = f"get_val_in_{arg_idx}(idx)"
            else:
                arg_name = f"texture(in_{arg_idx}, uv).r"
            self.var_map[nid] = arg_name
            return arg_name

        if op_type == "Constant":
            val = node.attributes.get("value", 0.0)
            res_var = f"v_{nid.replace('-', '_')}"
            self.var_map[nid] = res_var
            self.body_lines.append(f"  float {res_var} = {val};")
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
            self.body_lines.append(f"  float {res_var} = {expr};")
        elif op_type in ("Exp", "Log"):
            func_name = op_type.lower()
            self.body_lines.append(f"  float {res_var} = {func_name}({in_vars_mapped[0]});")
        elif op_type in ("Negative", "Neg"):
            self.body_lines.append(f"  float {res_var} = -{in_vars_mapped[0]};")
        else:
            args_str = ", ".join(in_vars_mapped)
            self.body_lines.append(f"  float {res_var} = {op_type.lower()}({args_str});")

        return res_var

    def _check_has_ndim_gt_1(self) -> bool:
        """Check if any input node in the graph has more than one dimension.

        Returns:
            bool: True if at least one input node has > 1 dimension, False otherwise.
        """
        for node in self.sorted_nodes:
            shape, _ = self._get_shape_and_strides(node)
            if len(shape) > 1:
                return True
        return False

    def _declare_glsl_inputs(self, input_nodes: list[Any], glsl_lines: list[str]) -> None:
        """Declare GLSL uniform sampler inputs and update the variable map.

        Args:
            input_nodes (list[Any]): List of input IR nodes.
            glsl_lines (list[str]): List of GLSL code lines to append to.
        """
        for idx, node in enumerate(input_nodes):
            glsl_lines.append(f"uniform sampler2D in_{idx};")
            nid = getattr(node, "id", "")
            shape, _ = self._get_shape_and_strides(node)
            if len(shape) > 1:
                self.var_map[nid] = f"get_val_in_{idx}(idx)"
            else:
                self.var_map[nid] = f"texture(in_{idx}, uv).r"

    def _compute_total_size(self, output_ids: list[str]) -> int:
        """Evaluate _compute_total_size operation.

        Args:
        output_ids (object): The output_ids parameter.

        Returns:
        int: Result.
        """
        total_size = 1
        if output_ids:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == output_ids[0]), None)
            if out_node:
                shape, _ = self._get_shape_and_strides(out_node)
                for dim in shape:
                    total_size *= dim
        return total_size

    def _build_js_shader_compiler(self, js_code: list[str], glsl_code_str: str, tex_width: int, tex_height: int) -> None:
        """Generate JS code to initialize WebGL context, compile shaders, and link the program.

        Args:
            js_code (list[str]): List of JS code lines to append to.
            glsl_code_str (str): The GLSL fragment shader source code.
            tex_width (int): The width of the WebGL canvas and textures.
            tex_height (int): The height of the WebGL canvas and textures.
        """
        js_code.append("// WebGL2 JavaScript Orchestrator Code Generated by ml-switcheroo-compiler")
        js_code.append("const fragmentShaderSource = `")
        js_code.append(glsl_code_str)
        js_code.append("`;")
        js_code.append("")
        js_code.append("const vertexShaderSource = `#version 300 es")
        js_code.append("in vec2 position;")
        js_code.append("void main(void) {")
        js_code.append("  gl_Position = vec4(position, 0.0, 1.0);")
        js_code.append("}`;")
        js_code.append("")
        js_code.append("function runWebGL(inputs) {")
        js_code.append("  const canvas = document.createElement('canvas');")
        js_code.append(f"  canvas.width = {tex_width};")
        js_code.append(f"  canvas.height = {tex_height};")
        js_code.append("  const gl = canvas.getContext('webgl2');")
        js_code.append("  if (!gl) {")
        js_code.append("    throw new Error('WebGL2 is not supported on this browser.');")
        js_code.append("  }")
        js_code.append("  gl.getExtension('EXT_color_buffer_float');")
        js_code.append("")
        js_code.append("  function compileShader(source, type) {")
        js_code.append("    const shader = gl.createShader(type);")
        js_code.append("    gl.shaderSource(shader, source);")
        js_code.append("    gl.compileShader(shader);")
        js_code.append("    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {")
        js_code.append("      throw new Error(gl.getShaderInfoLog(shader));")
        js_code.append("    }")
        js_code.append("    return shader;")
        js_code.append("  }")
        js_code.append("")
        js_code.append("  const vs = compileShader(vertexShaderSource, gl.VERTEX_SHADER);")
        js_code.append("  const fs = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);")
        js_code.append("  const program = gl.createProgram();")
        js_code.append("  gl.attachShader(program, vs);")
        js_code.append("  gl.attachShader(program, fs);")
        js_code.append("  gl.linkProgram(program);")
        js_code.append("  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {")
        js_code.append("    throw new Error(gl.getProgramInfoLog(program));")
        js_code.append("  }")
        js_code.append("  gl.useProgram(program);")
        js_code.append("")

    def _build_js_input_textures(self, input_nodes: list[Any], js_code: list[str]) -> None:
        """Generate JS code to bind inputs as WebGL textures.

        Args:
            input_nodes (list[Any]): List of input IR nodes.
            js_code (list[str]): List of JS code lines to append to.
        """
        import math

        js_code.append("  const positionBuffer = gl.createBuffer();")
        js_code.append("  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);")
        js_code.append("  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([")
        js_code.append("    -1, -1,  1, -1, -1,  1,")
        js_code.append("    -1,  1,  1, -1,  1,  1,")
        js_code.append("  ]), gl.STATIC_DRAW);")
        js_code.append("  const positionLoc = gl.getAttribLocation(program, 'position');")
        js_code.append("  gl.enableVertexAttribArray(positionLoc);")
        js_code.append("  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);")
        js_code.append("")

        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", f"n{idx}")
            shape, _ = self._get_shape_and_strides(node)
            num_elements = 1
            for d in shape:
                num_elements *= d
            w = int(math.ceil(math.sqrt(num_elements)))
            h = int(math.ceil(num_elements / w))
            js_code.append(f"  const tex_{idx} = gl.createTexture();")
            js_code.append(f"  gl.activeTexture(gl.TEXTURE0 + {idx});")
            js_code.append(f"  gl.bindTexture(gl.TEXTURE_2D, tex_{idx});")
            js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);")
            js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);")
            js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);")
            js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);")
            js_code.append(f"  let input_data_{idx} = inputs.{nid};")
            js_code.append(f"  if (input_data_{idx}.length < {w} * {h}) {{")
            js_code.append(f"    const padded = new Float32Array({w} * {h});")
            js_code.append(f"    padded.set(input_data_{idx});")
            js_code.append(f"    input_data_{idx} = padded;")
            js_code.append("  }")
            js_code.append(f"  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, {w}, {h}, 0, gl.RED, gl.FLOAT, input_data_{idx});")
            js_code.append(f"  gl.uniform1i(gl.getUniformLocation(program, 'in_{idx}'), {idx});")
            js_code.append("")

    def _build_js_framebuffer_and_render(self, js_code: list[str], tex_width: int, tex_height: int, total_size: int) -> None:
        """Generate JS code to configure the framebuffer, run the render pass, and read back the output.

        Args:
            js_code (list[str]): List of JS code lines to append to.
            tex_width (int): The width of the output texture.
            tex_height (int): The height of the output texture.
            total_size (int): The total number of elements to extract.
        """
        js_code.append("  const outTex = gl.createTexture();")
        js_code.append("  gl.bindTexture(gl.TEXTURE_2D, outTex);")
        js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);")
        js_code.append("  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);")
        js_code.append(f"  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, {tex_width}, {tex_height}, 0, gl.RGBA, gl.FLOAT, null);")
        js_code.append("")
        js_code.append("  const fb = gl.createFramebuffer();")
        js_code.append("  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);")
        js_code.append("  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);")
        js_code.append("")
        js_code.append(f"  gl.viewport(0, 0, {tex_width}, {tex_height});")
        js_code.append("  gl.drawArrays(gl.TRIANGLES, 0, 6);")
        js_code.append("")
        js_code.append(f"  const pixels = new Float32Array({tex_width} * {tex_height} * 4);")
        js_code.append(f"  gl.readPixels(0, 0, {tex_width}, {tex_height}, gl.RGBA, gl.FLOAT, pixels);")
        js_code.append("")
        js_code.append(f"  const outData = new Float32Array({total_size});")
        js_code.append(f"  for (let i = 0; i < {total_size}; i++) {{")
        js_code.append("    outData[i] = pixels[i * 4];")
        js_code.append("  }")
        js_code.append("")
        js_code.append("  return outData;")
        js_code.append("}")

    def _emit_glsl_main_coords(self, input_nodes: list[Any], has_ndim_gt_1: bool, glsl_lines: list[str]) -> None:
        """Emit GLSL code for deriving coordinate indices within the fragment shader.

        Args:
            input_nodes (list[Any]): List of input IR nodes.
            has_ndim_gt_1 (bool): Flag indicating if any input has > 1 dimension.
            glsl_lines (list[str]): List of GLSL code lines to append to.
        """
        if input_nodes:
            glsl_lines.append("  vec2 uv = gl_FragCoord.xy / vec2(textureSize(in_0, 0));")
            if has_ndim_gt_1:
                glsl_lines.append("  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * textureSize(in_0, 0).x;")
        else:
            glsl_lines.append("  vec2 uv = vec2(0.5, 0.5);")
            if has_ndim_gt_1:
                glsl_lines.append("  int idx = 0;")

    def _emit_glsl_main_body(self, output_ids: list[str], glsl_lines: list[str]) -> None:
        """Emit GLSL code for executing the operations and writing to the output fragment.

        Args:
            output_ids (list[str]): List of output node IDs.
            glsl_lines (list[str]): List of GLSL code lines to append to.
        """
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        for line in self.body_lines:
            glsl_lines.append(line)

        if output_ids:
            res_var = self.var_map.get(output_ids[0], output_ids[0])
            glsl_lines.append(f"  fragColor = vec4({res_var}, 0.0, 0.0, 1.0);")
        else:
            glsl_lines.append("  fragColor = vec4(0.0, 0.0, 0.0, 1.0);")

    def generate(self) -> str:
        """Generate WebGL ES 3.0 fragment shader code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable WebGL2 JavaScript orchestration code wrapper.
        """
        import math

        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids = getattr(self.graph, "outputs", []) or []

        self.code = []
        self.body_lines = []

        has_ndim_gt_1 = self._check_has_ndim_gt_1()

        glsl_lines = []
        glsl_lines.append("#version 300 es")
        glsl_lines.append("precision highp float;")
        glsl_lines.append("out vec4 fragColor;")

        self._declare_glsl_inputs(input_nodes, glsl_lines)

        tex_helpers = self._generate_texture_helpers(input_nodes)
        if tex_helpers:
            glsl_lines.append("")
            glsl_lines.append(tex_helpers)

        glsl_lines.append("")
        glsl_lines.append("void main() {")

        self._emit_glsl_main_coords(input_nodes, has_ndim_gt_1, glsl_lines)
        self._emit_glsl_main_body(output_ids, glsl_lines)

        glsl_lines.append("}")

        glsl_code_str = "\n".join(glsl_lines)

        total_size = self._compute_total_size(output_ids)

        tex_width = int(math.ceil(math.sqrt(total_size)))
        tex_height = int(math.ceil(total_size / tex_width))

        js_code = []
        self._build_js_shader_compiler(js_code, glsl_code_str, tex_width, tex_height)
        self._build_js_input_textures(input_nodes, js_code)
        self._build_js_framebuffer_and_render(js_code, tex_width, tex_height, total_size)

        return "\n".join(js_code)
