"""WebGL 2.0 Backend Emission."""

import os
from typing import Optional

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplatesConfig
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("edge_webgl")
class WebGLCodeGenerator(BaseGenerator):
    """WebGL 2.0 Code Generator for emitting fragment shader compute passes and browser JS orchestrator."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
        """Initialize WebGLCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines: list[str] = []

        yaml_path: str = os.path.join(os.path.dirname(__file__), "webgl_templates.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.config = WebglTemplatesConfig(**yaml.safe_load(f))
        else:
            self.config = WebglTemplatesConfig(templates={}, js_orchestration={})

    def _emit_node(self, node: "IRNode", input_map: dict[str, int], js: list[str]) -> tuple[int, int]:  # noqa: C901, PLR0912, PLR0915
        """Emit WebGL operations for a single node.

        Args:
            node (IRNode): The node to process.
            input_map (dict): The mapping of input node IDs to indices.
            js (list): The JS code list to append to.

        Returns:
            tuple[int, int]: The width and height of the emitted output.
        """
        op_type: str = getattr(node, "op_type", "")
        nid: str = getattr(node, "id", "")
        clean_id: str = nid.replace("-", "_")

        width: int = 32
        height: int = 32
        shape = getattr(node, "shape_metadata", None)
        if shape and len(shape) >= 2:
            height = int(shape[-2]) if isinstance(shape[-2], int) else 32
            width = int(shape[-1]) if isinstance(shape[-1], int) else 32
        elif shape and len(shape) == 1:
            height = 1
            width = int(shape[0]) if isinstance(shape[0], int) else 32

        template_config = self.config.templates.get(op_type.lower())
        if not template_config:
            raise ValueError(f"Missing WebGL shader template for operation: {op_type}")

        if isinstance(template_config, str):
            shader_body = template_config
            custom_setup = ""
        else:
            shader_body = template_config.body
            custom_setup = template_config.custom_setup

        escaped_shader: str = shader_body.replace("\n", "\\n").replace('"', '\\"')
        js.append(f'    const shader_{clean_id} = "{escaped_shader}";')

        js.append(f"    const prog_{clean_id} = createProgram(gl, vsSource, shader_{clean_id});")
        js.append(f"    let texOut_{clean_id} = createTexture(gl, null, {width}, {height});")
        js.append("    gl.bindFramebuffer(gl.FRAMEBUFFER, main_fbo);")
        js.append(f"    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texOut_{clean_id}, 0);")

        js.append(f"    gl.viewport(0, 0, {width}, {height});")
        js.append(f"    gl.useProgram(prog_{clean_id});")

        for i, in_id in enumerate(getattr(node, "inputs", [])):
            js.append(f"    gl.activeTexture(gl.TEXTURE{i});")
            if in_id in input_map:
                js.append(f"    gl.bindTexture(gl.TEXTURE_2D, inputs[{input_map[in_id]}]);")
            else:
                clean_in_id = in_id.replace("-", "_")
                js.append(f"    gl.bindTexture(gl.TEXTURE_2D, texOut_{clean_in_id});")

            uniform_names = ["A", "B", "C", "D"]
            if i < len(uniform_names):
                js.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, '{uniform_names[i]}'), {i});")

        if custom_setup:
            k_dim = 32
            node_inputs = getattr(node, "inputs", [])
            if node_inputs and len(node_inputs) > 0:
                in_node = getattr(self.graph, "nodes", {}).get(node_inputs[0])
                in_shape = getattr(in_node, "shape_metadata", None)
                if in_shape and len(in_shape) > 0 and isinstance(in_shape[-1], int):
                    k_dim = in_shape[-1]
            formatted_setup = custom_setup.format(clean_id=clean_id, width=width, height=height, k_dim=k_dim)
            for line in formatted_setup.strip().split("\n"):
                js.append(f"    {line}")

        js.append("    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);")
        js.append(f"    const aPos_{clean_id} = gl.getAttribLocation(prog_{clean_id}, 'aVertexPosition');")
        js.append(f"    gl.enableVertexAttribArray(aPos_{clean_id});")
        js.append(f"    gl.vertexAttribPointer(aPos_{clean_id}, 2, gl.FLOAT, false, 0, 0);")
        js.append("    gl.drawArrays(gl.TRIANGLES, 0, 6);")

        return width, height

    def generate(self) -> str:
        """Generate WebGL 2.0 compute orchestrator.

        Returns:
            str: Generated JS code.
        """
        js: list[str] = [self.config.js_orchestration.get("init", "")]
        js.append(self.config.js_orchestration.get("create_program", ""))
        js.append(self.config.js_orchestration.get("create_texture", ""))
        js.append(self.config.js_orchestration.get("read_pixels", ""))

        js.append("function evaluate_webgl(gl, inputs) {")
        js.append("    // Vertex shader for full screen quad")
        js.append("    const vsSource = `#version 300 es\\nin vec4 aVertexPosition;\\nvoid main() {\\n  gl_Position = aVertexPosition;\\n}`;")
        js.append("    // Shared FBO for ping-ponging")
        js.append("    const main_fbo = gl.createFramebuffer();")

        input_map: dict[str, int] = {}
        input_idx: int = 0
        for node in getattr(self.graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                input_map[node.id] = input_idx
                input_idx += 1

        last_width: int = 32
        last_height: int = 32

        for node in getattr(self.graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue

            last_width, last_height = self._emit_node(node, input_map, js)

        js.append(f"    return readPixels(gl, {last_width}, {last_height});")
        js.append("}")

        return "\n".join(js)
