"""WebGPU collective operations and structured control flow (WhileLoop, Cond, Scan)."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _node_attrs_dict,
)
from ml_switcheroo_compiler.backends.edge._webgpu_wgsl_templates import _WebGPUWgslTemplates
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.ir.core import ensure_graph


class _WebGPUCollectivesCF(_WebGPUWgslTemplates):
    """WebGPU visitor layer for collectives, WhileLoop, Cond, and Scan."""

    def visit_AllReduce(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllReduce.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllReduce", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllGather(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllGather.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllGather", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_AllToAll(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WebRTC AllToAll.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllToAll", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_ReduceScatter(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WebRTC ReduceScatter.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_Broadcast(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WebRTC Broadcast.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("Broadcast", "buf_in0_f32", op_id)
        return [f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}"], "1", "1", "1"

    def visit_WhileLoop(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Generate WGSL loop constructs via YAML templates with dynamic subgraph lowering.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))  # noqa: F841
        nelem = int(coerce_int(kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template = get_wgsl_template("while_loop")
        attrs = _node_attrs_dict(node)

        # Lower body subgraph
        body_graph = attrs.get("body")
        loop_body_lines: list[str] = []
        current_state_var: str = "current_state"

        if body_graph:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(ensure_graph(body_graph)):
                op_type: str = coerce_str(adapt_backend_module(sub_node).get_symbol("op_type", ""))
                mapping: dict[str, str] = OPS_REGISTRY.get(op_type, {}).get("variants", {}).get("edge_wgsl", {})
                expr: str = mapping.get("expr", "")
                if expr:
                    # Very simple regex-less token replacement for subgraph inlining
                    val_expr: str = expr.replace("buf_in0_f32[in0_offset]", current_state_var).replace(
                        "buf_in1_f32[in1_offset]", "1.0"
                    )
                    temp_var: str = (
                        f"tmp_{coerce_str(adapt_backend_module(sub_node).get_symbol('id', '0')).replace('-', '_')}"
                    )
                    loop_body_lines.append(f"var {temp_var} = {val_expr};")
                    current_state_var = temp_var
            if loop_body_lines:
                loop_body_lines.append(f"current_state = {current_state_var};")

        loop_body: str = (
            "\n    ".join(loop_body_lines) if loop_body_lines else "current_state = current_state + buf_in1_f32[idx];"
        )

        # Lower cond subgraph
        cond_graph = attrs.get("cond")
        condition_expr: str = str(
            attrs.get("condition_expr")
            or (f"current_state {attrs.get('comparator', '<')} {attrs.get('threshold', '10.0')}")
        )
        if cond_graph and "condition_expr" not in attrs:
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            comparator: str = "<"
            threshold: str = "10.0"
            for sub_node in topological_sort(ensure_graph(cond_graph)):
                op_type = coerce_str(adapt_backend_module(sub_node).get_symbol("op_type", ""))
                if op_type in ("Less", "Lt"):
                    comparator = "<"
                elif op_type in ("Greater", "Gt"):
                    comparator = ">"
                if "threshold" in _node_attrs_dict(sub_node):
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

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(64, 1, 1)"],
        )
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_Cond(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> tuple[list[str], str, str, str]:
        """Generate WGSL if/else block branching via YAML templates with dynamic subgraph lowering.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))  # noqa: F841
        nelem = int(coerce_int(kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        template = get_wgsl_template("cond")
        attrs = _node_attrs_dict(node)

        def _lower_branch(branch_graph: BackendArg, default_val: str) -> str:
            """Lower a branch subgraph into WGSL strings.

            Args:
                branch_graph (BackendArg): The subgraph to lower.
                default_val (str): The default input value string.

            Returns:
                str: Generated WGSL code string.
            """
            if not branch_graph:
                return f"buf_out_f32[idx] = {default_val};"
            lines: list[str] = []
            current_state: str = "buf_in0_f32[idx]"
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            for sub_node in topological_sort(ensure_graph(branch_graph)):
                op_type: str = coerce_str(adapt_backend_module(sub_node).get_symbol("op_type", ""))
                mapping: dict[str, str] = OPS_REGISTRY.get(op_type, {}).get("variants", {}).get("edge_wgsl", {})
                expr: str = mapping.get("expr", "")
                if expr:
                    val_expr: str = expr.replace("buf_in0_f32[in0_offset]", current_state).replace(
                        "buf_in1_f32[in1_offset]", "buf_in1_f32[idx]"
                    )
                    temp_var: str = (
                        f"tmp_{coerce_str(adapt_backend_module(sub_node).get_symbol('id', '0')).replace('-', '_')}"
                    )
                    lines.append(f"var {temp_var} = {val_expr};")
                    current_state = temp_var
            lines.append(f"buf_out_f32[idx] = {current_state};")
            return "\n    ".join(lines)

        true_body: str = _lower_branch(attrs.get("then_branch"), "buf_in1_f32[idx]")
        false_body: str = _lower_branch(attrs.get("else_branch"), "buf_in2_f32[idx]")

        condition_expr: str = str(
            attrs.get("condition_expr")
            or (f"buf_in0_f32[idx] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}")
        )
        body: str = str(template["body"]).format(
            nelem=nelem, condition_expr=condition_expr, true_body=true_body, false_body=false_body
        )

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(64, 1, 1)"],
        )
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"

    def visit_Scan(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> tuple[list[str], str, str, str]:
        """Generate WGSL scan via YAML templates.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))  # noqa: F841
        nelem = int(coerce_int(kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
            get_wgsl_template,
        )  # noqa: F401

        template = get_wgsl_template("scan")

        init_val: str = str(attrs.get("init_val", "0.0")) if (attrs := _node_attrs_dict(node)) else "0.0"
        scan_op_expr: str = str(attrs.get("scan_op_expr") or "acc + buf_in0_f32[i]")
        body: str = str(template["body"]).format(nelem=nelem, init_val=init_val, scan_op_expr=scan_op_expr)

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(64, 1, 1)"],
        )
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({nelem} / 64)", "1", "1"
