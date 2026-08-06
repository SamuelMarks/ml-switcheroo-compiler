# ruff: noqa: ANN401, C901, PLR0912, PLR0915
"""LLVM / C++ code generator for CPU fallback."""

from __future__ import annotations

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

    def generate(self, graph: Any = None) -> str:
        """Generate C++ code from an IR graph.

        Args:
            graph (Any): The logical graph to convert.

        Returns:
            str: The generated C++ source code.
        """
        self.lines = [
            "#include <iostream>",
            "#include <vector>",
            "#include <cmath>",
        ]

        if self.use_openmp:
            self.lines.append("#include <omp.h>")

        self.lines.append("")
        self.lines.append("void compute_graph() {")

        graph_to_use = graph if graph is not None else self.graph
        for _, node in graph_to_use.nodes.items():
            self._visit_node(node)

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit_binary_op(self, node: Any, op_sym: str) -> None:
        """Visit a binary operation node and emit C++ code.

        Args:
            node (Any): The binary operation node.
            op_sym (str): The binary operator symbol (e.g., '+', '-').
        """
        assert len(node.inputs) >= 2
        in1, in2 = node.inputs[:2]
        self.lines.append(f"    std::vector<float> {node.id}({in1}.size());")
        self.lines.append(f"    for(size_t i=0; i<{in1}.size(); ++i) {{")
        self.lines.append(f"        {node.id}[i] = {in1}[i] {op_sym} {in2}[i];")
        self.lines.append("    }")

    def _visit_unary_op(self, node: Any, func: str) -> None:
        """Visit a unary operation node and emit C++ code.

        Args:
            node (Any): The unary operation node.
            func (str): The unary function or operator symbol (e.g., '-', 'std::exp').
        """
        assert len(node.inputs) >= 1
        in1 = node.inputs[0]
        self.lines.append(f"    std::vector<float> {node.id}({in1}.size());")
        self.lines.append(f"    for(size_t i=0; i<{in1}.size(); ++i) {{")
        if func == "-":
            self.lines.append(f"        {node.id}[i] = -{in1}[i];")
        else:
            self.lines.append(f"        {node.id}[i] = {func}({in1}[i]);")
        self.lines.append("    }")

    def _visit_if_op(self, node: Any) -> None:
        """Visit a conditional (If) node and emit C++ code.

        Args:
            node (Any): The conditional operation node.
        """
        assert len(node.inputs) >= 1
        cond_var = node.inputs[0]
        self.lines.append(f"    if ({cond_var}[0] > 0.0f) {{")
        then_graph = node.attributes.get("then_branch")
        if then_graph:
            for _, sub_node in then_graph.nodes.items():
                self._visit_node(sub_node)
        self.lines.append("    } else {")
        else_graph = node.attributes.get("else_branch")
        if else_graph:
            for _, sub_node in else_graph.nodes.items():
                self._visit_node(sub_node)
        self.lines.append("    }")

    def _visit_loop_op(self, node: Any) -> None:
        """Visit a loop operation node and emit C++ code.

        Args:
            node (Any): The loop operation node.
        """
        self.lines.append("    while (true) {")
        cond_graph = node.attributes.get("cond")
        if cond_graph:
            for _, sub_node in cond_graph.nodes.items():
                self._visit_node(sub_node)
            self.lines.append("        // if (!cond_output) break;")
        body_graph = node.attributes.get("body")
        if body_graph:
            for _, sub_node in body_graph.nodes.items():
                self._visit_node(sub_node)
        self.lines.append("        break; // Prevent infinite loop in placeholder")
        self.lines.append("    }")

    def _visit_node(self, node: Any) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (Any): The IR node to visit.

        Raises:
            NotImplementedError: If the operator is not supported.
        """
        op = node.op_type

        binary_ops = {
            "Add": "+",
            "Subtract": "-",
            "Multiply": "*",
            "TrueDivide": "/",
            "Div": "/",
        }
        unary_ops = {
            "Exp": "std::exp",
            "Log": "std::log",
            "Negative": "-",
            "Neg": "-",
        }

        if op == "Input":
            # For simplicity, we just declare a vector
            self.lines.append(f"    std::vector<float> {node.id}; // Input")
        elif op == "Constant":
            val = node.attributes.get("value", 0.0)
            self.lines.append(f"    float {node.id} = {val}; // Constant")
        elif op in binary_ops:
            self._visit_binary_op(node, binary_ops[op])
        elif op in unary_ops:
            self._visit_unary_op(node, unary_ops[op])
        elif op == "MatMul":
            assert len(node.inputs) >= 2
            in1, in2 = node.inputs[:2]
            self.lines.append(f"    std::vector<float> {node.id}; // MatMul of {in1} and {in2}")
            self.lines.append("    // TODO: Need shape info for proper C++ GEMM.")
        elif op in ("If", "Cond"):
            self._visit_if_op(node)
        elif op in ("Loop", "WhileLoop"):
            self._visit_loop_op(node)
        elif op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
        else:
            raise NotImplementedError(f"Operator {op} is not supported in CppGenerator.")

    def compile(self, code: str) -> Any:
        """Compile the generated code.

        Args:
            code (str): The C++ source code.

        Returns:
            Any: A callable that executes the code (placeholder).
        """

        def executable() -> str:
            """Simulate the execution of the compiled C++ code.

            Returns:
                str: A simulated execution result.
            """
            return "Execution simulated"

        return executable

    def execute(self, graph: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the graph using the C++ generator.

        Args:
            graph (Any): The logical graph to execute.
            *args: Arguments for execution.
            **kwargs: Keyword arguments for execution.

        Returns:
            Any: The result of execution.
        """
        code = self.generate(graph)
        executable = self.compile(code)
        return executable()
