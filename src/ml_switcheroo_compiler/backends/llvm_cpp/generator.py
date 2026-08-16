"""Module generator.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""LLVM / C++ code generator for CPU fallback."""


from typing import Any

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("llvm_cpp")
class CppGenerator(BaseGenerator):
    """C++ backend generator."""

    def __init__(self, graph: Any = None, use_simd: bool = True, use_openmp: bool = True) -> None:
        """Initialize the C++ generator.

        Args:
            graph (Any): The graph parameter.
            use_simd (bool): The use_simd parameter.
            use_openmp (bool): The use_openmp parameter.
        """
        super().__init__(graph=graph)
        self.use_simd = use_simd
        self.use_openmp = use_openmp
        self.lines: list[str] = []

    def _get_shape(self, node: Any) -> list[int]:
        """_get_shape function.

        Args:
        self (Any): The self parameter.
        node (Any): The node parameter.

        Returns:
        Any: Result.
        """
        shape = getattr(node, "shape_metadata", None)
        if not shape:
            return [1]
        if isinstance(shape, (int, float)):
            return [int(shape)]
        res = []
        for s in shape:
            res.append(int(s))
        return res

    def _num_elements(self, shape: list[int]) -> int:
        """_num_elements function.

        Args:
        self (Any): The self parameter.
        shape (Any): The shape parameter.

        Returns:
        Any: Result.
        """
        n = 1
        for s in shape:
            n *= s
        return n

    def _get_strides(self, shape: list[int]) -> list[int]:
        """_get_strides function.

        Args:
        self (Any): The self parameter.
        shape (Any): The shape parameter.

        Returns:
        Any: Result.
        """
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def generate(self, graph: Any = None) -> str:
        """Generate C++ code from an IR graph.

        Args:
            graph (Any): The logical graph to convert.

        Returns:
            str: The generated C++ source code.
        """
        import os

        import yaml

        tmpl_path = os.path.join(os.path.dirname(__file__), "cpp_templates.yaml")
        with open(tmpl_path) as f:
            data = yaml.safe_load(f)
        prelude = data.get("prelude", "")
        self.lines = prelude.strip().split("\n")

        self.lines.append("void compute_graph() {")

        graph_to_use = graph if graph is not None else self.graph
        for _, node in graph_to_use.nodes.items():
            self._visit_node(node, graph_to_use)

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit_if_op(self, node: Any, graph_to_use: Any = None) -> None:
        """_visit_if_op function.

        Args:
        self (Any): The self parameter.
        node (Any): The node parameter.
        graph_to_use (Any): The graph_to_use parameter.

        Returns:
        Any: Result.
        """
        assert len(node.inputs) >= 1
        cond_var = node.inputs[0]
        self.lines.append(f"    if ({cond_var}.data[0] > 0.0f) {{")
        then_graph = node.attributes.get("then_branch")
        if then_graph:
            for _, sub_node in then_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("    } else {")
        else_graph = node.attributes.get("else_branch")
        if else_graph:
            for _, sub_node in else_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("    }")

    def _visit_loop_op(self, node: Any, graph_to_use: Any = None) -> None:
        """_visit_loop_op function.

        Args:
        self (Any): The self parameter.
        node (Any): The node parameter.
        graph_to_use (Any): The graph_to_use parameter.

        Returns:
        Any: Result.
        """
        self.lines.append("    while (true) {")
        cond_graph = node.attributes.get("cond")
        if cond_graph:
            for _, sub_node in cond_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        body_graph = node.attributes.get("body")
        if body_graph:
            for _, sub_node in body_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("        break;")
        self.lines.append("    }")

    def _visit_node(self, node: Any, graph_to_use: Any = None) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (Any): The IR node to visit.
            graph_to_use (Any): The graph.
        """
        op = node.op_type

        if op == "Input":
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}); // Input")
        elif op == "Constant":
            val = node.attributes.get("value", 0.0)
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {val};")
        elif op in ("If", "Cond"):
            self._visit_if_op(node, graph_to_use)
        elif op in ("Loop", "WhileLoop"):
            self._visit_loop_op(node, graph_to_use)
        elif op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
        else:
            from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template
            from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

            op_def = OPS_REGISTRY.get(op, {})
            mapping = op_def.get("variants", {}).get("llvm_cpp", {})

            if not mapping:
                out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
                self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}); // Fallback Unimplemented {op}")
            else:
                template = get_cpp_template(mapping["template"])

                in0_node = graph_to_use.nodes.get(node.inputs[0]) if len(node.inputs) > 0 else None
                in0_shape = self._get_shape(in0_node) if in0_node else [1, 1]
                out_shape = self._get_shape(node)
                out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"

                M = out_shape[0] if len(out_shape) > 0 else 1
                N = out_shape[1] if len(out_shape) > 1 else 1
                K = in0_shape[1] if len(in0_shape) > 1 else 1

                expr_format_args = {"clean_id": node.id, "out_shape_str": out_shape_str, "in0": node.inputs[0] if len(node.inputs) > 0 else "dummy", "in1": node.inputs[1] if len(node.inputs) > 1 else "dummy", "rank": len(out_shape), "M": M, "N": N, "K": K}
                expr_format_args.update(mapping)

                body = template["body"].format(**expr_format_args)
                for line in body.split("\n"):
                    if line.strip():
                        self.lines.append(f"    {line}")

    def compile(self, code: str) -> Any:
        """Compile the generated code."""

        def executable() -> str:
            """Executable function.

            Returns:
            Any: Result.
            """
            return "Execution simulated (compiled)"

        return executable

    def execute(self, graph: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the graph using the C++ generator."""
        code = self.generate(graph)
        executable = self.compile(code)
        return executable()
