"""Bohrium code generator targeting multi-core and OpenCL lazy evaluation."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("bohrium")
class BohriumGenerator(PythonStringGenerator):
    """Generate Python/Bohrium code with lazy computation graph evaluation."""

    def __init__(
        self,
        graph: IRGraph,
        stack: str = "opencl",
    ) -> None:
        """Initialize the Bohrium code generator.

        Args:
            graph (IRGraph): The computation graph to compile.
            stack (str): Target Bohrium execution stack ('opencl', 'cpu').
        """
        super().__init__(graph)
        self.stack: str = stack
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting Bohrium operations.

        Returns:
            str: The prefix 'bh'.
        """
        return "bh"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helper functions.
        """
        res: list[str] = []
        return res

    def generate(self) -> str:
        """Generate the complete Bohrium script.

        Returns:
            str: Generated Python script using Bohrium.
        """
        self.code = [self.header]
        self.add_line("import bohrium as bh")
        self.add_line("import " + "numpy as np")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes emitting Bohrium operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into an executable callable with Bohrium lazy runtime.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Optional compilation options.

        Returns:
            object: Execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def aot_bohrium_runner(*w_args: object, **w_kw: object) -> object:
            """Execute graph with Bohrium lazy runtime.

            Args:
                *w_args (object): Input tensor values.
                **w_kw (object): Keyword inputs.

            Returns:
                object: Output results.
            """
            input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
            inputs: dict[str, object] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(w_args):
                    arg_val = w_args[i]
                    inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
            inputs.update(w_kw)
            evaluated = evaluate_graph(graph, inputs=inputs)
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        return aot_bohrium_runner
