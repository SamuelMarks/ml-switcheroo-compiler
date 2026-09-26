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
            input_map (dict[str, int]): The mapping of input node IDs to indices.
            js (list[str]): The JS code list to append to.

        Returns:
            tuple[int, int]: The width and height of the emitted output.

        Raises:
            ValueError: If shader template is missing for the operation.
        """
        op_type: str = getattr(node, "op_type", "")
        nid: str = getattr(node, "id", "")
        clean_id: str = nid.replace("-", "_")

        shape = getattr(node, "shape_metadata", None)
        node_inputs = getattr(node, "inputs", []) or []
        in_shape = None
        if node_inputs and len(node_inputs) > 0:
            in_node = getattr(self.graph, "nodes", {}).get(node_inputs[0])
            in_shape = getattr(in_node, "shape_metadata", None) if in_node else None

        eff_shape = shape if (shape and len(shape) > 0) else (in_shape if (in_shape and len(in_shape) > 0) else None)

        width: int = 32
        height: int = 32
        if eff_shape and len(eff_shape) == 1:
            height = 1
            width = int(eff_shape[0]) if isinstance(eff_shape[0], int) else 32
        elif eff_shape and len(eff_shape) == 2:
            height = int(eff_shape[0]) if isinstance(eff_shape[0], int) else 32
            width = int(eff_shape[1]) if isinstance(eff_shape[1], int) else 32
        elif eff_shape and len(eff_shape) > 2:
            if op_type.lower() in ("conv2d", "maxpool2d", "avgpool2d"):
                outer_spatial = 1
                for d in eff_shape[:-1]:
                    if isinstance(d, int):
                        outer_spatial *= d
                height = max(1, outer_spatial)
                width = int(eff_shape[-1]) if isinstance(eff_shape[-1], int) else 32
            else:
                outer = 1
                for d in eff_shape[:-1]:
                    if isinstance(d, int):
                        outer *= d
                height = max(1, outer)
                width = int(eff_shape[-1]) if isinstance(eff_shape[-1], int) else 32

        norm_op: str = op_type.lower()
        norm_no_underscore: str = norm_op.replace("_", "")
        template_config = self.config.templates.get(norm_op) or self.config.templates.get(norm_no_underscore) or next((v for k, v in self.config.templates.items() if k.replace("_", "") == norm_no_underscore), None)
        if not template_config:
            raise ValueError(f"Missing WebGL shader template for operation: {op_type}")

        if isinstance(template_config, str):
            shader_body = template_config
            custom_setup = ""
        else:
            shader_body = template_config.body
            custom_setup = template_config.custom_setup

        escaped_shader: str = shader_body.replace("\n", "\\n").replace('"', '\\"')

        binding_lines: list[str] = []
        for i, in_id in enumerate(getattr(node, "inputs", [])):
            binding_lines.append(f"    gl.activeTexture(gl.TEXTURE{i});")
            if in_id in input_map:
                binding_lines.append(f"    gl.bindTexture(gl.TEXTURE_2D, inputs[{input_map[in_id]}]);")
            else:
                clean_in_id = in_id.replace("-", "_")
                binding_lines.append(f"    gl.bindTexture(gl.TEXTURE_2D, texOut_{clean_in_id});")

            uniform_names = ["A", "B", "C", "D"]
            if i < len(uniform_names):
                binding_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, '{uniform_names[i]}'), {i});")

        setup_lines: list[str] = []
        if custom_setup:
            k_dim: int = 32
            in_h: int = 32
            in_w: int = 32
            k_h: int = 3
            k_w: int = 3
            node_inputs = getattr(node, "inputs", [])
            if node_inputs and len(node_inputs) > 0:
                in_node = getattr(self.graph, "nodes", {}).get(node_inputs[0])
                in_shape = getattr(in_node, "shape_metadata", None)
                if in_shape and len(in_shape) > 0 and isinstance(in_shape[-1], int):
                    k_dim = in_shape[-1]
                    in_w = in_shape[-1]
                if in_shape and len(in_shape) >= 2 and isinstance(in_shape[-2], int):
                    in_h = in_shape[-2]
            if node_inputs and len(node_inputs) > 1:
                k_node = getattr(self.graph, "nodes", {}).get(node_inputs[1])
                k_shape = getattr(k_node, "shape_metadata", None)
                if k_shape and len(k_shape) >= 2:
                    if isinstance(k_shape[-1], int):
                        k_w = k_shape[-1]
                    if isinstance(k_shape[-2], int):
                        k_h = k_shape[-2]
            attrs: dict[str, object] = getattr(node, "attributes", {}) or {}
            starts = attrs.get("starts", attrs.get("start", [0, 0]))
            start_x: int = 0
            start_y: int = 0
            if isinstance(starts, (list, tuple)):
                if len(starts) == 1:
                    start_y = int(starts[0]) if isinstance(starts[0], int) else 0
                elif len(starts) >= 2:
                    start_y = int(starts[0]) if isinstance(starts[0], int) else 0
                    start_x = int(starts[1]) if isinstance(starts[1], int) else 0
            elif isinstance(starts, int):
                start_y = starts

            concat_axis: int = int(attrs.get("axis", 0)) if isinstance(attrs.get("axis", 0), int) else 0
            split_boundary: int = in_h if concat_axis == 0 else in_w
            reduce_axis: int = int(attrs.get("axis", 1)) if isinstance(attrs.get("axis", 1), int) else 1

            formatted_setup = custom_setup.format(
                clean_id=clean_id,
                width=width,
                height=height,
                k_dim=k_dim,
                in_h=in_h,
                in_w=in_w,
                k_h=k_h,
                k_w=k_w,
                start_x=start_x,
                start_y=start_y,
                concat_axis=concat_axis,
                split_boundary=split_boundary,
                reduce_axis=reduce_axis,
            )
            for line in formatted_setup.strip().split("\n"):
                if line.strip():
                    setup_lines.append(f"    {line}")

        if shape and len(shape) > 0:
            shape_ints: list[int] = [int(s) if isinstance(s, int) else 1 for s in shape]
            curr_stride: int = 1
            strides: list[int] = []
            for s in reversed(shape_ints):
                strides.insert(0, curr_stride)
                curr_stride *= s
            for i, st in enumerate(strides):
                setup_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, 'stride_{i}'), {st});")
            setup_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, 'rank'), {len(shape_ints)});")
            for i, s_dim in enumerate(shape_ints):
                setup_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, 'dim_{i}'), {s_dim});")
            setup_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, 'out_width'), {width});")
            setup_lines.append(f"    gl.uniform1i(gl.getUniformLocation(prog_{clean_id}, 'out_height'), {height});")

        node_pass_tpl: Optional[str] = self.config.js_orchestration.get("node_pass")
        if node_pass_tpl:
            rendered = node_pass_tpl.format(
                clean_id=clean_id,
                escaped_shader=escaped_shader,
                width=width,
                height=height,
                bindings="\n".join(binding_lines),
                setup="\n".join(setup_lines),
            )
            js.append(rendered)
        else:
            js.append(f'    const shader_{clean_id} = "{escaped_shader}";')
            js.append(f"    const prog_{clean_id} = createProgram(gl, vsSource, shader_{clean_id});")
            num_outputs: int = int(getattr(node, "attributes", {}).get("num_outputs", 1))
            js.append("    gl.bindFramebuffer(gl.FRAMEBUFFER, currentFbo);")
            if num_outputs > 1:
                draw_buffers: list[str] = []
                for out_i in range(num_outputs):
                    js.append(f"    let texOut_{clean_id}_{out_i} = createTexture(gl, null, {width}, {height});")
                    js.append(f"    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT{out_i}, gl.TEXTURE_2D, texOut_{clean_id}_{out_i}, 0);")
                    draw_buffers.append(f"gl.COLOR_ATTACHMENT{out_i}")
                js.append(f"    gl.drawBuffers([{', '.join(draw_buffers)}]);")
            else:
                js.append(f"    let texOut_{clean_id} = createTexture(gl, null, {width}, {height});")
                js.append(f"    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texOut_{clean_id}, 0);")
            js.append(f"    gl.viewport(0, 0, {width}, {height});")
            js.append(f"    gl.useProgram(prog_{clean_id});")
            js.extend(binding_lines)
            js.extend(setup_lines)
            js.append("    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);")
            js.append(f"    const aPos_{clean_id} = gl.getAttribLocation(prog_{clean_id}, 'aVertexPosition');")
            js.append(f"    gl.enableVertexAttribArray(aPos_{clean_id});")
            js.append(f"    gl.vertexAttribPointer(aPos_{clean_id}, 2, gl.FLOAT, false, 0, 0);")
            js.append("    gl.drawArrays(gl.TRIANGLES, 0, 6);")
            js.append("    currentFbo = (currentFbo === main_fbo) ? fboPong : main_fbo;")

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
        js.append("    // Ping-pong framebuffers for multi-pass compute without stalls")
        js.append("    const main_fbo = gl.createFramebuffer();")
        js.append("    const fboPong = gl.createFramebuffer();")
        js.append("    let currentFbo = main_fbo;")

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

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into ahead-of-time WebGL executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (object): Compiler options.

        Returns:
            object: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact

        js_source = self.generate()

        def runner(*args: object, **runner_kwargs: object) -> object:
            """Execute compiled WebGL artifact using fallback graph evaluation.

            Args:
                *args (object): Input tensor arguments.
                **runner_kwargs (object): Optional execution options.

            Returns:
                object: Output tensor or tuple of output tensors.
            """
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            feed_dict: dict[str, object] = {}
            for idx, arg in enumerate(args):
                if idx < len(graph.inputs):
                    feed_dict[graph.inputs[idx]] = getattr(arg, "data", arg)
            res_dict = evaluate_graph(graph, feed_dict)
            if not graph.outputs:
                return res_dict
            if len(graph.outputs) == 1:
                return res_dict.get(graph.outputs[0])
            return tuple(res_dict.get(out) for out in graph.outputs)

        shaders_map = {node_id: f"shader_{node_id}" for node_id, node in getattr(graph, "nodes", {}).items() if getattr(node, "op_type", "").lower() not in ("input", "output")}

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=js_source.encode("utf-8"),
            source_code=js_source,
            metadata={"backend": "webgl", "shaders": shaders_map, "options": kwargs},
        )
