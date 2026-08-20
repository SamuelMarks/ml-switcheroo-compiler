"""WebGL 2.0 Backend Emission."""

import os
from typing import Any, Optional

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplatesConfig
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("edge_webgl")
class WebGLCodeGenerator(BaseGenerator):
    """WebGL 2.0 Code Generator for emitting fragment shader compute passes and browser JS orchestrator."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
        """Initialize WebGLCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines: list[str] = []

        yaml_path = os.path.join(os.path.dirname(__file__), "webgl_templates.yaml")
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
        js = [self.config.js_orchestration.get("init", "")]
        js.append(self.config.js_orchestration.get("create_texture", ""))

        for node in getattr(self.graph, "nodes", {}).values():
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")

            # Emit standard fragment shader
            shader_tpl = self.config.templates.get(op_type.lower(), "")
            if shader_tpl:
                # Use standard string replace instead of backticks in template
                escaped_shader = shader_tpl.replace("\n", "\\n").replace('"', '\\"')
                js.append(f'const shader_{clean_id} = "{escaped_shader}";')

        return "\n".join(js)
