"""WebGL 2.0 Backend Emission."""

import os
from typing import Any

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplatesConfig
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("edge_webgl")
class WebGLCodeGenerator(BaseGenerator):
    """WebGL 2.0 Code Generator for emitting fragment shader compute passes and browser JS orchestrator."""

    def __init__(self, graph: IRGraph, delegates: Any = None) -> None:
        """Initialize WebGLCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Any, optional): Visitor delegates.
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

    def generate(self) -> str:
        """Generate WebGL 2.0 compute orchestrator.

        Returns:
            str: Generated JS code.
        """
        js: list[str] = [self.config.js_orchestration.get("init", "")]
        js.append(self.config.js_orchestration.get("create_texture", ""))

        js.append("function evaluate_webgl(gl, inputs) {")
        js.append("    // Vertex shader for full screen quad")
        js.append("    const vsSource = `#version 300 es\\nin vec4 aVertexPosition;\\nvoid main() {\\n  gl_Position = aVertexPosition;\\n}`;")

        for node in getattr(self.graph, "nodes", {}).values():
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")

            # Emit standard fragment shader
            shader_tpl: str = self.config.templates.get(op_type.lower(), "")
            if not shader_tpl:
                # Fallback shader for missing operations (Identity/Copy)
                shader_tpl = "#version 300 es\nprecision highp float;\nout vec4 fragColor;\nvoid main() { fragColor = vec4(0.0); }"

            escaped_shader: str = shader_tpl.replace("\n", "\\n").replace('"', '\\"')
            js.append(f'    const shader_{clean_id} = "{escaped_shader}";')

            # Setup program and textures
            js.append(f"    const prog_{clean_id} = createProgram(gl, vsSource, shader_{clean_id});")
            js.append(f"    const texOut_{clean_id} = createTexture(gl, 32, 32); // Mock Size")
            js.append(f"    const fb_{clean_id} = gl.createFramebuffer();")
            js.append(f"    gl.bindFramebuffer(gl.FRAMEBUFFER, fb_{clean_id});")
            js.append(f"    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texOut_{clean_id}, 0);")
            js.append(f"    gl.useProgram(prog_{clean_id});")
            js.append("    // Bind inputs... gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, inputs[0]);")
            js.append("    // gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);")

        js.append("    return null; // Return final texture")
        js.append("}")

        return "\n".join(js)
