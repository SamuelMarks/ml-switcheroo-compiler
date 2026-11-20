"""Structured control flow (WhileLoop, Cond, Scan) and MatMul visitors for WASM."""

from __future__ import annotations

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    WasmAttrType,
    _node_attrs,
    _str_list_attr,
)
from ml_switcheroo_compiler.backends.edge._wasm_conv_collectives import _WasmConvCollectives
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.ir.core import ensure_graph


class _WasmControlMatMul(_WasmConvCollectives):
    """WASM emission layer for WhileLoop, Loop, Cond, If, Scan, and MatMul."""

    def visit_WhileLoop(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
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
        attrs: dict[str, WasmAttrType] = _node_attrs(node)

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty loop body"
        if body_graph:
            import ml_switcheroo_compiler.backends.edge.wasm as _wasm_mod

            subgen = _wasm_mod.WasmCodeGenerator(ensure_graph(body_graph))
            loop_body = subgen.generate()

        condition_expr: str = str(
            attrs.get("condition_expr")
            or (f"buf_{inputs[0]}[0] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}" if inputs else "1")
        )
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

    def visit_Loop(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
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

    def visit_Cond(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Cond with dynamic subgraph lowering.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("cond")
        attrs: dict[str, WasmAttrType] = _node_attrs(node)

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

            import ml_switcheroo_compiler.backends.edge.wasm as _wasm_mod

            subgen = _wasm_mod.WasmCodeGenerator(ensure_graph(branch_graph))

            # Map inputs for the subgraph generator to point to the parent's input buffers
            # Offset by 1 because inputs[0] is the conditional predicate
            input_remap = {}
            for i, in_name in enumerate(_str_list_attr(branch_graph, "inputs")):
                parent_idx = i + 1 if (i + 1) < len(inputs) else 0
                input_remap[in_name] = inputs[parent_idx] if inputs else "dummy"

            # Overload the input mapping logic internally
            subgen.var_map.update(input_remap)

            out_code = subgen.generate()

            # Final output mapping
            bg_outputs = _str_list_attr(branch_graph, "outputs")
            if bg_outputs:
                out_id = subgen.var_map.get(bg_outputs[0], bg_outputs[0].replace("-", "_"))
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{out_id}[i];"
            else:
                out_code += f"\\n  for(int i=0; i<{nelem}; i++) buf_{clean_id}[i] = buf_{default_input}[i];"

            return out_code

        # Check for branch_graphs array first (standardized layout)
        raw_bg = attrs.get("branch_graphs", [])
        branch_graphs: list[WasmAttrType] = list(raw_bg) if isinstance(raw_bg, (list, tuple)) else []
        if branch_graphs:
            true_graph: WasmAttrType = branch_graphs[0] if len(branch_graphs) > 0 else None
            false_graph: WasmAttrType = branch_graphs[1] if len(branch_graphs) > 1 else None
        else:
            true_graph = attrs.get("then_branch")
            false_graph = attrs.get("else_branch")

        true_body: str = _lower_branch(true_graph, inputs[1] if len(inputs) > 1 else inputs[0] if inputs else "dummy")
        false_body: str = _lower_branch(false_graph, inputs[2] if len(inputs) > 2 else inputs[0] if inputs else "dummy")

        cond_expr: str = str(
            attrs.get("condition_expr")
            or (f"buf_{inputs[0]}[0] {attrs.get('comparator', '>')} {attrs.get('threshold', '0.0')}" if inputs else "1")
        )
        body: str = template["body"].format(condition_expr=cond_expr, true_body=true_body, false_body=false_body)
        for line in body.split("\n"):
            self.add_line(line)

    def visit_If(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate If (alias for Cond).

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self.visit_Cond(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Scan(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Scan using recursive subgraph execution.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("scan")
        attrs: dict[str, WasmAttrType] = _node_attrs(node)

        body_graph: WasmAttrType = attrs.get("body_graph") or attrs.get("body")
        loop_body: str = "// Empty scan body"
        if body_graph:
            import ml_switcheroo_compiler.backends.edge.wasm as _wasm_mod

            subgen = _wasm_mod.WasmCodeGenerator(ensure_graph(body_graph))
            loop_body = subgen.generate()

        init_val: str = str(attrs.get("init_val", "0.0"))
        scan_op_expr: str = str(
            attrs.get("scan_op_expr")
            or (
                f"buf_{clean_id}[i] = buf_{inputs[0]}[i];\\n    " + loop_body.replace("\\n", "\\n    ")
                if inputs
                else f"buf_{clean_id}[i] = 1.0f;"
            )
        )
        body: str = template["body"].format(
            clean_id=clean_id, nelem=nelem, init_val=init_val, scan_op_expr=scan_op_expr
        )
        for line in body.split("\n"):
            self.add_line(line)

    def visit_MatMul(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate MatMul WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = _node_attrs(node)
        if attrs.get("tiling"):
            template: dict[str, str] = get_wasm_template("tiled_matmul")
            # For simplicity, fetch the heuristic from the environment or use defaults
            TILE_M = 4
            TILE_N = 4
            TILE_K = 4
            in0: str = inputs[0] if len(inputs) > 0 else "dummy"
            in1: str = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if len(shape) > 0 else 1
            N = shape[1] if len(shape) > 1 else 1

            in0_node: WasmAttrType = (
                next(
                    (
                        n
                        for n in self.sorted_nodes
                        if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                    ),
                    None,
                )
                if len(inputs) > 0
                else None
            )
            in0_shape = adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

            body: str = template["body"].format(
                M=M, N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, in0=in0, in1=in1
            )
            for line in body.split("\n"):
                self.add_line(line)
        else:
            # fallback
            template = get_wasm_template("matmul")
            in0 = inputs[0] if len(inputs) > 0 else "dummy"
            in1 = inputs[1] if len(inputs) > 1 else "dummy"
            M = shape[0] if len(shape) > 0 else 1
            N = shape[1] if len(shape) > 1 else 1

            in0_node = (
                next(
                    (
                        n
                        for n in self.sorted_nodes
                        if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                    ),
                    None,
                )
                if len(inputs) > 0
                else None
            )
            in0_shape = adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else [1, 1]
            K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1
            body = template["body"].format(M=M, N=N, K=K, in0=in0, in1=in1, clean_id=clean_id)
            for line in body.split("\n"):
                self.add_line(line)
