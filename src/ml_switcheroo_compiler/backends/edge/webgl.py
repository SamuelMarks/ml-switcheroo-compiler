# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

    def _get_shape_and_strides(self, node: Any) -> tuple[list[int], list[int]]:
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

    def generic_visit(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
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

    def _get_templates(self) -> Any:
        """Load WebGL templates.

        Returns:
            Any: The templates config.
        """
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplatesConfig

        path = os.path.join(os.path.dirname(__file__), "webgl_templates.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
            return WebglTemplatesConfig(**data)

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

        self.code.clear()
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

        tex_width = max(1, int(math.ceil(math.sqrt(total_size))))
        tex_height = int(math.ceil(total_size / tex_width))

        js_code = []  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        templates = self._get_templates()
        js_code.append(templates.js_shader_compiler.format(glsl_code_str=glsl_code_str, tex_width=tex_width, tex_height=tex_height))
        js_code.append(templates.js_input_textures_setup)

        for idx, node in enumerate(input_nodes):
            nid = getattr(node, "id", f"n{idx}")
            shape, _ = self._get_shape_and_strides(node)
            num_elements = 1
            for d in shape:
                num_elements *= d
            if num_elements < 1:
                num_elements = 1
            w = int(math.ceil(math.sqrt(num_elements)))
            h = int(math.ceil(num_elements / w))
            js_code.append(templates.js_input_texture_bind.format(idx=idx, nid=nid, w=w, h=h))

        js_code.append(templates.js_framebuffer_and_render.format(tex_width=tex_width, tex_height=tex_height, total_size=total_size))

        return "\n".join(js_code)
