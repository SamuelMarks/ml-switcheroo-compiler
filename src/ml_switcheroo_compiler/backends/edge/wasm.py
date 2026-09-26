# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

import os
from typing import Optional, Union, cast

import yaml

WasmAttrType = Union["IRNode", "IRGraph", dict[str, Union[int, float, str, bool, list, tuple, dict, None]], list, str, int, float, tuple, bool, None]

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class WasmCodeGenerator(BaseGenerator):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code."""

    def __init__(self, graph: IRGraph, delegates: WasmAttrType = None) -> None:
        """Initialize WasmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (WasmAttrType, optional): Visitor delegates.
        """
        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import (
            OFFLINE_DIAGNOSTIC_OPS,
            strip_offline_diagnostic_nodes_pass,
        )

        sanitized_graph = strip_offline_diagnostic_nodes_pass(graph)
        super().__init__(sanitized_graph, delegates)
        self.sorted_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") not in OFFLINE_DIAGNOSTIC_OPS]
        self.var_map: dict[str, str] = {}
        self.is_simd: bool = False

    def _allocate_aligned_memory(self, size_bytes: int, alignment: int = 16) -> str:
        """_allocate_aligned_memory function.

        Args:
        self: The self parameter.
        size_bytes (int): The size_bytes parameter.
        alignment (int): The alignment parameter.

        Returns:
        str: Result.
        """
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        """_generate_striding_logic function.

        Args:
        self: The self parameter.
        shape (list[int]): The shape parameter.

        Returns:
        tuple[list[int], str]: Result.
        """
        if not shape:
            return [], "0"
        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code: str = " + ".join([f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)])
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        """_map_type function.

        Args:
        self: The self parameter.
        dtype (str): The dtype parameter.

        Returns:
        str: Result.
        """
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "bool": "bool",
        }.get(str(dtype).lower(), "float")

    def _num_elements(self, shape: list[int]) -> int:
        """_num_elements function.

        Args:
        self: The self parameter.
        shape (list[int]): The shape parameter.

        Returns:
        int: Result.
        """
        if isinstance(shape, (int, float)):
            return 1
        n: int = 1
        for s in shape:
            n *= s
        return n

    def get_helper_functions(self) -> list[str]:
        """Return C++ helper functions/macros for missing WASM math.

        Returns:
            list[str]: Lines of C++ macros/functions.
        """
        helpers: list[str] = ["// --- Scalar Fallback Helpers ---"]
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()

                # Load scalar helpers
                scalars: dict[str, str] = data.get("scalars") or {}
                for name, body in scalars.items():
                    helpers.append(f"inline float _scalar_{name.lower()}(float a, float b) {{ {body} }}")

                helpers.append("// --- SIMD Fast Math Approximations from YAML ---")

                intrinsics: dict[str, dict[str, str]] = data.get("intrinsics", {})
                for _op, intrinsic_data in intrinsics.items():
                    if intrinsic_data.get("macro_name") and intrinsic_data.get("simd_expr"):
                        is_binary: bool = "b" in intrinsic_data["simd_expr"]
                        arg_sig: str = "v128_t a, v128_t b" if is_binary else "v128_t x"
                        helpers.append(f"inline v128_t {intrinsic_data['macro_name']}({arg_sig}) {{")
                        for line in intrinsic_data["simd_expr"].split("\n"):
                            if line.strip():
                                helpers.append(f"    {line}")
                        helpers.append("}")

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers

        helpers.extend(get_cpp_helpers())
        return helpers

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: WasmAttrType) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (WasmAttrType): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return getattr(node, "id", "")

    def visit_Conv2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Conv2D WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        # Handle Folded BatchNorm Math Transformation
        if attrs.get("folded_batch_norm"):
            bn_inputs: list[str] = attrs.get("bn_inputs", [])
            if len(bn_inputs) == 4:
                scale, bias, mean, var = [self.var_map.get(inp, inp) for inp in bn_inputs]
                eps: float = float(attrs.get("epsilon", 1e-5))
                # We emit scalar C++ for folding
                self.add_line(f"// Folded BatchNorm for {clean_id}")
                self.add_line(f"float mult_{clean_id} = {scale} / std::sqrt({var} + {eps});")
                # This assumes scalar inputs for simplicity in WASM mapping, or we'd map this as a loop over elements

        if attrs.get("tiling"):
            template: dict[str, str] = get_wasm_template("im2col_conv2d")
        else:
            template = get_wasm_template("conv2d")

        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[WasmAttrType] = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs_list]
        in0_shape: list[int] = getattr(input_nodes[0], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        w_shape: list[int] = getattr(input_nodes[1], "shape_metadata", [1, 1, 1, 1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1]

        if not in0_shape:
            in0_shape = [1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [1, 1, 1, int(in0_shape)]
        else:
            in0_shape = list(in0_shape)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if not w_shape:
            w_shape = [1, 1, 1, 1]
        elif isinstance(w_shape, (int, float)):
            w_shape = [1, 1, 1, int(w_shape)]
        else:
            w_shape = list(w_shape)
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if not shape:
            shape = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape = [1, 1, 1, int(shape)]
        else:
            shape = list(shape)
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs = getattr(node, "attributes", {})
        stride: WasmAttrType = attrs.get("stride", 1)
        stride_h: int = stride[0] if isinstance(stride, (tuple, list)) else stride
        stride_w: int = stride[1] if isinstance(stride, (tuple, list)) else stride

        expr_args: dict[str, WasmAttrType] = {
            "B": shape[0],
            "out_channels": shape[1],
            "out_height": shape[2],
            "out_width": shape[3],
            "in_channels": in0_shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "filter_h": w_shape[2],
            "filter_w": w_shape[3],
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
        }
        body: str = template["body"].format(**expr_args)

        lines: list[str] = []
        for line in body.split("\n"):
            lines.append(f"    {line}")
        for line in lines:
            self.add_line(line)

    def visit_Conv3D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Conv3D WASM SIMD.

        Args:
            node (IRNode): The IR node representing Conv3D.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier string for output buffer.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Target output shape dimensions.
            nelem (int): Total number of output elements.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Conv3D")

        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[WasmAttrType] = [next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None) for inp in inputs_list]
        in0_shape: list[int] = getattr(input_nodes[0], "shape_metadata", [1, 1, 1, 1, 1]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        w_shape: list[int] = getattr(input_nodes[1], "shape_metadata", [1, 1, 1, 1, 1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1, 1]

        if not in0_shape:
            in0_shape = [1, 1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [1, 1, 1, 1, int(in0_shape)]
        else:
            in0_shape = list(in0_shape)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape

        if not w_shape:
            w_shape = [1, 1, 1, 1, 1]
        elif isinstance(w_shape, (int, float)):
            w_shape = [1, 1, 1, 1, int(w_shape)]
        else:
            w_shape = list(w_shape)
        if len(w_shape) < 5:
            w_shape = [1] * (5 - len(w_shape)) + w_shape

        if not shape:
            shape = [1, 1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape = [1, 1, 1, 1, int(shape)]
        else:
            shape = list(shape)
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {}) or {}
        stride: WasmAttrType = attrs.get("stride", attrs.get("strides", 1))
        if isinstance(stride, (tuple, list)):
            if len(stride) == 3:
                stride_d, stride_h, stride_w = int(stride[0]), int(stride[1]), int(stride[2])
            elif len(stride) == 2:
                stride_d, stride_h, stride_w = 1, int(stride[0]), int(stride[1])
            else:
                s_val = int(stride[0]) if stride else 1
                stride_d = stride_h = stride_w = s_val
        else:
            stride_d = stride_h = stride_w = int(stride)

        pad: WasmAttrType = attrs.get("padding", attrs.get("pad", 0))
        if isinstance(pad, (tuple, list)):
            if len(pad) == 3:
                pad_d, pad_h, pad_w = int(pad[0]), int(pad[1]), int(pad[2])
            elif len(pad) == 2:
                pad_d, pad_h, pad_w = 0, int(pad[0]), int(pad[1])
            else:
                p_val = int(pad[0]) if pad else 0
                pad_d = pad_h = pad_w = p_val
        elif isinstance(pad, str) and pad.upper() == "SAME":
            pad_d = max(0, (shape[2] - 1) * stride_d + w_shape[2] - in0_shape[2]) // 2
            pad_h = max(0, (shape[3] - 1) * stride_h + w_shape[3] - in0_shape[3]) // 2
            pad_w = max(0, (shape[4] - 1) * stride_w + w_shape[4] - in0_shape[4]) // 2
        else:
            pad_d = pad_h = pad_w = int(pad) if isinstance(pad, (int, float)) else 0

        dilation: WasmAttrType = attrs.get("dilation", attrs.get("dilations", 1))
        if isinstance(dilation, (tuple, list)):
            if len(dilation) == 3:
                dilation_d, dilation_h, dilation_w = int(dilation[0]), int(dilation[1]), int(dilation[2])
            else:
                d_val = int(dilation[0]) if dilation else 1
                dilation_d = dilation_h = dilation_w = d_val
        else:
            dilation_d = dilation_h = dilation_w = int(dilation)

        expr_args: dict[str, WasmAttrType] = {
            "B": shape[0],
            "out_channels": shape[1],
            "out_depth": shape[2],
            "out_height": shape[3],
            "out_width": shape[4],
            "in_channels": in0_shape[1],
            "in_depth": in0_shape[2],
            "in_height": in0_shape[3],
            "in_width": in0_shape[4],
            "filter_d": w_shape[2],
            "filter_h": w_shape[3],
            "filter_w": w_shape[4],
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_d": pad_d,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_d": dilation_d,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.add_line(f"    {line}")

    def visit_AllReduce(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllReduce."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllReduce", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllGather(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllGather."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllGather", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllToAll(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC AllToAll."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("AllToAll", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_ReduceScatter(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC ReduceScatter."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_Broadcast(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Emit WebRTC Broadcast."""
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = getattr(node, "id", "op")
        js_code: str = emit_webrtc_op("Broadcast", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_WhileLoop(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate WhileLoop with loop option annotations.

        Args:
            node (IRNode): The IR node representing the loop.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier for generated code variables.
            inputs (list[str]): Input variable names.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("while_loop")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty loop body"
        if body_graph:
            subgen = WasmCodeGenerator(body_graph)
            loop_body = subgen.generate()

        condition_expr: str = str(attrs.get("condition_expr") or (f"buf_{inputs[0]}[0] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}" if inputs else "1"))
        max_iters = attrs.get("maximum_iterations")
        if max_iters is None:
            max_iters = attrs.get("max_iters", 10)
        parallel_iters = attrs.get("parallel_iterations")
        swap_memory = attrs.get("swap_memory")
        shape_invariants = attrs.get("shape_invariants")

        annotations: list[str] = []
        if parallel_iters is not None:
            annotations.append(f"parallel_iterations={parallel_iters}")
        if swap_memory is not None:
            annotations.append(f"swap_memory={swap_memory}")
        if max_iters is not None:
            annotations.append(f"maximum_iterations={max_iters}")
        if shape_invariants is not None:
            annotations.append(f"shape_invariants={shape_invariants}")
        if annotations:
            self.add_line(f"  // Loop annotations: {', '.join(annotations)}")

        body: str = template["body"].format(
            in0=inputs[0] if inputs else "dummy",
            clean_id=clean_id,
            condition_expr=condition_expr,
            max_iters=max_iters,
            loop_body=loop_body,
        )
        for line in body.split("\n"):
            self.add_line(line)

    def visit_Loop(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Loop (alias for WhileLoop).

        Args:
            node (IRNode): The IR node representing the loop.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier for generated code variables.
            inputs (list[str]): Input variable names.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self.visit_WhileLoop(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Cond(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Cond with dynamic subgraph lowering."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("cond")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        def _lower_branch(branch_graph: WasmAttrType, default_input: str) -> str:
            """Lower a branch subgraph into C++ strings.

            Args:
                branch_graph (WasmAttrType): The subgraph to lower.
                default_input (str): The default input buffer name.

            Returns:
                str: Generated C++ code string.
            """
            if not branch_graph:
                return f"for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

            subgen = WasmCodeGenerator(branch_graph)

            # Map inputs for the subgraph generator to point to the parent's input buffers
            # Offset by 1 because inputs[0] is the conditional predicate
            input_remap = {}
            for i, in_name in enumerate(getattr(branch_graph, "inputs", [])):
                parent_idx = i + 1 if (i + 1) < len(inputs) else 0
                input_remap[in_name] = inputs[parent_idx] if inputs else "dummy"

            # Overload the input mapping logic internally
            subgen.var_map.update(input_remap)

            out_code = subgen.generate()

            # Final output mapping
            if branch_graph.outputs:
                out_id = subgen.var_map.get(branch_graph.outputs[0], branch_graph.outputs[0].replace("-", "_"))
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{out_id}[i];"
            else:
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

            return out_code

        # Check for branch_graphs array first (standardized layout)
        branch_graphs: list[WasmAttrType] = attrs.get("branch_graphs", [])
        if branch_graphs:
            true_graph: WasmAttrType = branch_graphs[0] if len(branch_graphs) > 0 else None
            false_graph: WasmAttrType = branch_graphs[1] if len(branch_graphs) > 1 else None
        else:
            true_graph = attrs.get("then_branch")
            false_graph = attrs.get("else_branch")

        true_body: str = _lower_branch(true_graph, inputs[1] if len(inputs) > 1 else inputs[0] if inputs else "dummy")
        false_body: str = _lower_branch(false_graph, inputs[2] if len(inputs) > 2 else inputs[0] if inputs else "dummy")

        cond_expr: str = str(attrs.get("condition_expr") or (f"buf_{inputs[0]}[0] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}" if inputs else "1"))
        body: str = template["body"].format(condition_expr=cond_expr, true_body=true_body, false_body=false_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_If(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate If (alias for Cond)."""
        self.visit_Cond(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Scan(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Scan using recursive subgraph execution."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("scan")
        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty scan body"
        if body_graph:
            subgen = WasmCodeGenerator(body_graph)
            loop_body = subgen.generate()

        init_val: str = str(attrs.get("init_val", "0.0"))
        scan_op_expr: str = str(attrs.get("scan_op_expr") or (f"buf_{clean_id}[i] = buf_{inputs[0]}[i];\\n    " + loop_body.replace("\\n", "\\n    ") if inputs else f"buf_{clean_id}[i] = 1.0f;"))
        body: str = template["body"].format(clean_id=clean_id, nelem=nelem, init_val=init_val, scan_op_expr=scan_op_expr)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_MatMul(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MatMul WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = getattr(node, "attributes", {})
        if attrs.get("tiling"):
            template: dict[str, str] = get_wasm_template("tiled_matmul")
            # For simplicity, fetch the heuristic from the environment or use defaults
            TILE_M = 4
            TILE_N = 4
            TILE_K = 4
            in0: str = inputs[0] if len(inputs) > 0 else "dummy"
            in1: str = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: list[int] = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

            body: str = template["body"].format(M=M, N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, in0=in0, in1=in1)
            for line in body.split("\n"):
                self.add_line(line)
        else:
            # fallback
            template = get_wasm_template("matmul")
            in0 = inputs[0] if len(inputs) > 0 else "dummy"
            in1 = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

            in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            body = template["body"].format(M=M, N=N, K=K, in0=in0, in1=in1, clean_id=clean_id)
            for line in body.split("\n"):
                self.add_line(line)

    def visit_MaxPool2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MaxPool2D WASM."""
        self._generate_pooling2d(node, clean_id, inputs, shape, "max_pool2d")

    def visit_AvgPool2D(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate AvgPool2D WASM."""
        self._generate_pooling2d(node, clean_id, inputs, shape, "avg_pool2d")

    def _generate_pooling2d(self, node: IRNode, clean_id: str, inputs: list[str], shape: list[int], template_name: str) -> None:
        """Generate Pooling 2D."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template(template_name)
        attrs = getattr(node, "attributes", {})

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "") == inputs[0]), None) if len(inputs) > 0 else None

        in0_shape = getattr(in0_node, "shape_metadata", [1, 1, 1, 1]) if in0_node else [1, 1, 1, 1]

        if not in0_shape:
            in0_shape = [1, 1, 1, 1]
        elif isinstance(in0_shape, (int, float)):
            in0_shape = [1, 1, 1, int(in0_shape)]
        else:
            in0_shape = list(in0_shape)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if not shape:
            shape = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape = [1, 1, 1, int(shape)]
        else:
            shape = list(shape)
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        kernel_size = attrs.get("kernel_size", attrs.get("pool_size", 1))
        kernel_h = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        kernel_w = kernel_size[1] if isinstance(kernel_size, (list, tuple)) else kernel_size

        stride = attrs.get("stride", attrs.get("strides", 1))
        stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
        stride_w = stride[1] if isinstance(stride, (list, tuple)) else stride

        expr_args = {
            "B": shape[0],
            "C": shape[1],
            "out_height": shape[2],
            "out_width": shape[3],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
        }
        body = template.get("body", "").format(**expr_args)
        for line in body.split("\\n"):
            self.add_line(f"    {line}")

    def visit_Linear(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Linear (Dense) WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Linear")

        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
        N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        has_bias = "true" if in2 != "dummy" else "false"
        body = template.get("body", "").format(M=M, N=N, K=K, in0=in0, in1=in1, in2=in2, clean_id=clean_id, nelem=nelem, has_bias=has_bias)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Attention(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Attention WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Attention")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        seq_len = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        embed_dim = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, seq_len=seq_len, embed_dim=embed_dim)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_MaxPool(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate MaxPool WASM."""
        self.visit_MaxPool2D(node, op_type, clean_id, inputs, shape, nelem)

    def visit_LayerNorm(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate LayerNorm WASM."""
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("LayerNorm")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"  # scale
        in2 = inputs[2] if len(inputs) > 2 else "dummy"  # bias

        attrs = getattr(node, "attributes", {})
        eps = float(attrs.get("epsilon", 1e-5))

        in0_node = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
        rows = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        cols = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, rows=rows, cols=cols, eps=eps)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Trig(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Trig (Sin/Cos) WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Exp(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Exp WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Log(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Log WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Tanh(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Tanh WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sigmoid(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Sigmoid WASM."""
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_ReduceSum(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate ReduceSum WASM.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("ReduceSum")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Add(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Add WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sub(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Sub WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Mul(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Mul WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Div(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Div WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Min(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Min WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Max(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Max WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Abs(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Abs WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Neg(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Neg WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sqrt(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Sqrt WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Relu(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Relu WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Ceil(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Ceil WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Floor(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Floor WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Round(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Round WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sin(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Sin WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Cos(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate Cos WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def _generate_binary_simd_op(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate vector unrolled v128 binary operations with remainder loop peeling and bounds checking.

        Args:
            node (IRNode): The IR node.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier for generated code variables.
            inputs (list[str]): Input variable names.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        in0: str = inputs[0] if len(inputs) > 0 else "dummy"
        in1: str = inputs[1] if len(inputs) > 1 else in0

        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        simd_macro: str = ""
        scalar_expr: str = ""

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
                intrinsics_dict = data.get("intrinsics", {})
                intr = intrinsics_dict.get(op_type) or next((v for k, v in intrinsics_dict.items() if k.lower() == op_type.lower()), None)
                if intr:
                    simd_macro = intr.get("macro_name", "")
                    if intr.get("scalar_fallback"):
                        scalar_expr = intr["scalar_fallback"].format(f"buf_{in0}[i_{clean_id}]", f"buf_{in1}[i_{clean_id}]")
                if not scalar_expr:
                    scalars_dict = data.get("scalars", {})
                    if op_type.lower() in scalars_dict:
                        scalar_expr = f"_scalar_{op_type.lower()}(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])"
        if not scalar_expr:
            binary_scalars: dict[str, str] = {
                "add": f"buf_{in0}[i_{clean_id}] + buf_{in1}[i_{clean_id}]",
                "sub": f"buf_{in0}[i_{clean_id}] - buf_{in1}[i_{clean_id}]",
                "mul": f"buf_{in0}[i_{clean_id}] * buf_{in1}[i_{clean_id}]",
                "div": f"buf_{in0}[i_{clean_id}] / buf_{in1}[i_{clean_id}]",
                "min": f"std::min(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])",
                "max": f"std::max(buf_{in0}[i_{clean_id}], buf_{in1}[i_{clean_id}])",
            }
            scalar_expr = binary_scalars.get(op_type.lower(), f"buf_{in0}[i_{clean_id}]")

        self.add_line("  // Bounds check")
        self.add_line(f"  if ({nelem} > size) return; // Out of bounds")

        if simd_macro:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} <= {nelem} - 4; i_{clean_id} += 4) {{")
            self.add_line(f"      v128_t vec_a = wasm_v128_load(&buf_{in0}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_b = wasm_v128_load(&buf_{in1}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_out = {simd_macro}(vec_a, vec_b);")
            self.add_line(f"      wasm_v128_store(&buf_{clean_id}[i_{clean_id}], vec_out);")
            self.add_line("  }")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")
        else:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")

    def _generate_vector_unrolled_op(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """Generate vector unrolled v128 operations with remainder loop peeling and bounds checking."""
        in_id = inputs[0] if inputs else "dummy"

        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        simd_macro = ""
        scalar_expr = ""

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
                intrinsics_dict = data.get("intrinsics", {})
                intr = intrinsics_dict.get(op_type) or next((v for k, v in intrinsics_dict.items() if k.lower() == op_type.lower()), None)
                if intr:
                    simd_macro = intr.get("macro_name", "")
                    if intr.get("scalar_fallback"):
                        scalar_expr = intr["scalar_fallback"].format(f"buf_{in_id}[i_{clean_id}]")
                if not scalar_expr:
                    scalars_dict = data.get("scalars", {})
                    scalar_def = scalars_dict.get(op_type.lower())
                    if scalar_def:
                        body_str = scalar_def.replace("return ", "").rstrip(";")
                        scalar_expr = body_str.replace("a", f"buf_{in_id}[i_{clean_id}]")
        else:
            builtins = {
                "sin": "std::sin({0})",
                "cos": "std::cos({0})",
                "exp": "std::exp({0})",
                "log": "std::log({0})",
                "tanh": "std::tanh({0})",
                "sqrt": "std::sqrt({0})",
                "abs": "std::abs({0})",
                "neg": "-{0}",
                "relu": "std::max(0.0f, {0})",
            }
            if op_type.lower() in builtins:
                scalar_expr = builtins[op_type.lower()].format(f"buf_{in_id}[i_{clean_id}]")

        if not simd_macro and not scalar_expr:
            raise UnimplementedMathError(f"Operation '{op_type}' lacks WASM SIMD intrinsics and scalar fallback.")

        if simd_macro and not scalar_expr:
            scalar_expr = f"wasm_f32x4_extract_lane({simd_macro}(wasm_f32x4_splat(buf_{in_id}[i_{clean_id}])), 0)"

        self.add_line("  // Bounds check")
        self.add_line(f"  if ({nelem} > size) return; // Out of bounds")

        if simd_macro:
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} <= {nelem} - 4; i_{clean_id} += 4) {{")
            self.add_line(f"      v128_t vec_in = wasm_v128_load(&buf_{in_id}[i_{clean_id}]);")
            self.add_line(f"      v128_t vec_out = {simd_macro}(vec_in);")
            self.add_line(f"      wasm_v128_store(&buf_{clean_id}[i_{clean_id}], vec_out);")
            self.add_line("  }")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")
        else:
            # Scalar fallback
            self.add_line(f"  int i_{clean_id} = 0;")
            self.add_line(f"  for(; i_{clean_id} < {nelem}; i_{clean_id}++) {{")
            self.add_line(f"      buf_{clean_id}[i_{clean_id}] = {scalar_expr};")
            self.add_line("  }")

    def _generate_op(self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int) -> None:
        """_generate_op function.

        Args:
        self: The self parameter.
        node (IRNode): The node parameter.
        op_type (str): The op_type parameter.
        clean_id (str): The clean_id parameter.
        inputs (list[str]): The inputs parameter.
        shape (list[int]): The shape parameter.
        nelem (int): The nelem parameter.

        Returns:
        None: Result.
        """
        if hasattr(self, f"visit_{op_type}"):
            getattr(self, f"visit_{op_type}")(node, op_type, clean_id, inputs, shape, nelem)
            return

        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import OFFLINE_DIAGNOSTIC_OPS

        if op_type in OFFLINE_DIAGNOSTIC_OPS:
            return

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template
        from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

        op_def: dict[str, WasmAttrType] = OPS_REGISTRY.get(op_type, {})
        mapping: dict[str, WasmAttrType] = op_def.get("variants", {}).get("edge_wasm_simd", {})

        if not mapping:
            import os

            import yaml

            from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

            yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
            is_intrinsic = False
            if os.path.exists(yaml_path):
                with open(yaml_path) as f:
                    idata = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()
                    is_intrinsic = op_type in idata.get("intrinsics", {}) or op_type.lower() in idata.get("scalars", {}) or op_type.lower() in [k.lower() for k in idata.get("intrinsics", {})]

            if is_intrinsic:
                if len(inputs) <= 1:
                    self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)
                else:
                    self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)
                return

            raise UnimplementedMathError(f"Missing WASM SIMD template for {op_type}")
        else:
            template: dict[str, str] = get_wasm_template(mapping["template"])
            if "body" not in template:
                print(f"DEBUG: op_type={op_type}, mapping={mapping}, template={template}")

            # Get dimensions
            in0_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", "").replace("-", "_") == inputs[0]), None) if len(inputs) > 0 else None
            in0_shape: list[int] = getattr(in0_node, "shape_metadata", None) if in0_node else [1, 1]
            if not in0_shape:
                in0_shape = [1, 1]
            elif isinstance(in0_shape, (int, float)):
                in0_shape = [int(in0_shape)]

            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            N = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 1
            M = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1

            expr_format_args: dict[str, WasmAttrType] = {
                "nelem": nelem,
                "clean_id": clean_id,
                "op_type": op_type,
                "in0": inputs[0] if len(inputs) > 0 else "dummy",
                "in1": inputs[1] if len(inputs) > 1 else "dummy",
                "K": K,
                "N": N,
                "M": M,
                "nelem_in": getattr(node, "inputs_nelem", [1])[0],
            }
            expr_format_args.update(mapping)

            if "body" not in template:
                raise UnimplementedMathError(f"MISSING BODY FOR: {op_type} template: {template}")
            body: str = template["body"].format(**expr_format_args)
            for line in body.split("\n"):
                if line.strip():
                    self.add_line(f"  {line}")

    def generate(self) -> str:
        """Generate WASM-compatible, highly optimized C++ source code with WASM v128 SIMD and scalar peeling.

        Returns:
            str: Generated highly vectorizable C++ kernel code with remainder loops.
        """
        input_nodes: list[IRNode] = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        output_ids: list[str] = getattr(self.graph, "outputs", []) or []

        self.code.clear()
        func_params: list[str] = []
        for idx, node in enumerate(input_nodes):
            meta_dtype: str = self._map_type(getattr(node, "dtype", "float32"))
            func_params.append(f"const {meta_dtype}* __restrict__ in_{idx}")

        for i, out_id in enumerate(output_ids):
            out_node: WasmAttrType = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            meta_dtype = self._map_type(getattr(out_node, "dtype", "float32")) if out_node else "float"
            func_params.append(f"{meta_dtype}* __restrict__ out_{i}")

        func_params.append("int size")
        params_str: str = ", ".join(func_params)

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        headers_tpl: str = get_wasm_template("kernel_headers").get("body", "")
        for line in headers_tpl.strip().split("\n"):
            self.add_line(line)
        self.add_line("")
        for helper in self.get_helper_functions():
            self.add_line(helper)
        self.add_line("")

        main_start_tpl: str = get_wasm_template("kernel_main_start").get("body", "")
        formatted_start: str = main_start_tpl.format(params_str=params_str)
        for line in formatted_start.strip().split("\n"):
            self.add_line(line)

        self.add_line("  // Buffer arenas")
        arenas: dict[int, int] = {}
        for node in self.sorted_nodes:
            arena_id: int = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset: int = getattr(node, "attributes", {}).get("buffer_offset", 0)
            shape: list[int] = getattr(node, "shape_metadata", None)
            nelem: int = self._num_elements(shape if shape else [1])
            if arena_id not in arenas:
                arenas[arena_id] = 0
            arenas[arena_id] = max(arenas[arena_id], offset + nelem * 4)

        for arena_id, total_size in arenas.items():
            self.add_line(f"  float* buf_arena_{arena_id} = (float*)std::aligned_alloc(16, {total_size});")

        self.add_line("  // Pointers and Input Copies")
        for idx, node in enumerate(input_nodes):
            nid: str = getattr(node, "id", "")
            clean_id: str = nid.replace("-", "_")
            arena_id = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            shape = getattr(node, "shape_metadata", None)
            nelem = self._num_elements(shape if shape else [1])
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")
            self.add_line(f"  std::copy(in_{idx}, in_{idx} + {nelem}, buf_{clean_id});")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                continue
            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            arena_id = getattr(node, "attributes", {}).get("buffer_id", 0)
            offset = getattr(node, "attributes", {}).get("buffer_offset", 0) // 4
            self.add_line(f"  float* buf_{clean_id} = buf_arena_{arena_id} + {offset};")

        self.add_line("  float dummy_val = 0.0f;")
        self.add_line("  float* buf_dummy = &dummy_val;")
        self.add_line("")
        self.add_line("  // Compute nodes sequentially")

        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import OFFLINE_DIAGNOSTIC_OPS

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type in ("Input", "Output", "Constant") or op_type in OFFLINE_DIAGNOSTIC_OPS:
                continue

            nid = getattr(node, "id", "")
            clean_id = nid.replace("-", "_")
            inputs: list[str] = [inp.replace("-", "_") for inp in getattr(node, "inputs", [])]
            shape = getattr(node, "shape_metadata", None)
            if not shape:
                shape = [1]
            elif isinstance(shape, (int, float)):
                shape = [int(shape)]
            nelem = self._num_elements(shape)

            self._generate_op(node, op_type, clean_id, inputs, shape, nelem)

        self.add_line("  // Copy Outputs")
        for i, out_id in enumerate(output_ids):
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            if out_node:
                clean_id = out_node.id.replace("-", "_")
                shape = getattr(out_node, "shape_metadata", None)
                nelem = self._num_elements(shape if shape else [1])
                self.add_line(f"  std::copy(buf_{clean_id}, buf_{clean_id} + {nelem}, out_{i});")

        self.add_line("  // Free temporary allocations")
        for arena_id in arenas:
            self.add_line(f"  std::free(buf_arena_{arena_id});")

        self.add_line("}")
        self.add_line("}")

        return "\n".join(self.code)

    def compile_wasm(self, output_dir: str = ".") -> Optional[tuple[str, str]]:
        """Automatically compile the generated C++ source code into WASM and JS binary files using emcc or clang.

        Args:
            output_dir (str): Directory where output files (.wasm and .js) should be saved.

        Returns:
            Optional[tuple[str, str]]: Paths to (js_loader_path, wasm_binary_path) if compilation succeeded, else None.
        """
        import os
        import shutil
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w") as temp_file:
            temp_file.write(self.generate())
            temp_file_path: str = temp_file.name

        try:
            emcc_bin = shutil.which("emcc")
            if emcc_bin:
                js_out: str = os.path.join(output_dir, "kernel.js")
                wasm_out: str = os.path.join(output_dir, "kernel.wasm")
                cmd: list[str] = [emcc_bin, "-O3", "-msimd128", "-s", "EXPORTED_FUNCTIONS=['_main_kernel']", "-s", "STANDALONE_WASM", temp_file_path, "-o", js_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            clang_bin = shutil.which("clang")
            if clang_bin:
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [clang_bin, "--target=wasm32", "-O3", "-msimd128", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=main_kernel", temp_file_path, "-o", wasm_out]
                subprocess.run(cmd, check=True, capture_output=True)
                return "", wasm_out

            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError("Neither emcc nor clang found for WASM compilation.")

        except subprocess.CalledProcessError as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            err_msg: str = e.stderr.decode("utf-8") if e.stderr else str(e)
            raise CompilationError(f"WASM compilation failed: {err_msg}") from e
        except Exception as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"WASM compilation failed with unknown error: {e}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> tuple[str, str]:
        """Compile IRGraph into a standalone .wasm binary via compile_wasm.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Compiler options.

        Returns:
            tuple[str, str]: Paths to generated js loader and wasm binary files.
        """
        if graph != self.graph:
            self.graph = graph
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            self.sorted_nodes = topological_sort(graph)

        out_dir = str(kwargs.get("output_dir", "."))
        res = self.compile_wasm(output_dir=out_dir)
        return res if res is not None else ("", "")

    def export_graph_payload(self) -> dict[str, Union[list[str], list[dict[str, Union[str, list[str], list[int]]]]]]:
        """Export serialized graph definition for browser-side WASM execution.

        Returns:
            dict[str, Union[list[str], list[dict[str, Union[str, list[str], list[int]]]]]]: Graph payload containing nodes, inputs, outputs, and shape metadata.
        """
        nodes_list: list[dict[str, Union[str, list[str], list[int]]]] = []
        for node in self.sorted_nodes:
            nodes_list.append(
                {
                    "id": str(getattr(node, "id", "")),
                    "op_type": str(getattr(node, "op_type", "")),
                    "inputs": [str(i) for i in getattr(node, "inputs", [])],
                    "shape_metadata": [int(s) for s in (getattr(node, "shape_metadata", ()) or ()) if isinstance(s, (int, float))],
                }
            )
        return {
            "inputs": [str(i) for i in getattr(self.graph, "inputs", [])],
            "outputs": [str(o) for o in getattr(self.graph, "outputs", [])],
            "nodes": nodes_list,
        }

    def apply_shape_telemetry(self, telemetry: dict[str, object]) -> None:
        """Update IRGraph node shape metadata and refine symbolic dimensions from client telemetry.

        Args:
            telemetry (dict[str, object]): Mapping or payload containing learned concrete shapes.
        """
        shape_data: dict[str, object] = telemetry.get("shapes", telemetry) if isinstance(telemetry, dict) else {}
        if not isinstance(shape_data, dict):
            return

        solved_env: dict[str, int] = {}
        for node in self.sorted_nodes:
            nid: str = getattr(node, "id", "")
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

        # Refine symbolic dimensions across all nodes in the graph
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

    def generate_wat(self) -> str:
        """Generate valid WebAssembly Text (WAT) module with SIMD-128 instructions.

        Returns:
            str: WAT module string suitable for wat2wasm compilation.
        """
        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "wasm_opcodes.yaml")
        op_map: dict[str, dict[str, object]] = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                if isinstance(raw, dict) and "opcodes" in raw and isinstance(raw["opcodes"], dict):
                    op_map = raw["opcodes"]

        nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") != "Input"]
        wat_lines: list[str] = [
            "(module",
            '  (memory (export "memory") 1)',
            '  (func (export "compute") (param $in i32) (param $len i32) (param $out i32)',
            "    (local $i i32)",
            "    (local.set $i (i32.const 0))",
            "    (block $B",
            "      (loop $L",
            "        (br_if $B (i32.ge_u (local.get $i) (local.get $len)))",
        ]

        for node in nodes:
            op_type = getattr(node, "op_type", "").lower()
            spec = op_map.get(op_type, {})
            simd_inst: str = str(spec.get("instruction", "f32x4.mul"))
            arity: int = int(spec.get("arity", 1 if simd_inst in ("f32x4.neg", "f32x4.sqrt", "f32x4.abs", "f32x4.ceil", "f32x4.floor", "f32x4.trunc", "f32x4.nearest") else 2))

            if arity == 1:
                wat_lines.extend(
                    [
                        "        (v128.store",
                        "          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))",
                        f"          ({simd_inst}",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "          )",
                        "        )",
                    ]
                )
            else:
                wat_lines.extend(
                    [
                        "        (v128.store",
                        "          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))",
                        f"          ({simd_inst}",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "          )",
                        "        )",
                    ]
                )

        wat_lines.extend(
            [
                "        (local.set $i (i32.add (local.get $i) (i32.const 4)))",
                "        (br $L)",
                "      )",
                "    )",
                "  )",
                ")",
            ]
        )
        return "\n".join(wat_lines)
