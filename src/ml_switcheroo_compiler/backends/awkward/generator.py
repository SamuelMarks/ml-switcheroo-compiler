"""Awkward Array code generator supporting ragged and nested dimension structures."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("awkward")
class AwkwardGenerator(PythonStringGenerator):
    """Generate Python/Awkward code operating on nested and ragged array structures."""

    def __init__(
        self,
        graph: IRGraph,
        ragged_mode: bool = True,
    ) -> None:
        """Initialize the Awkward code generator.

        Args:
            graph (IRGraph): The computation graph to compile.
            ragged_mode (bool): Whether to preserve ragged nested structures.
        """
        super().__init__(graph)
        self.ragged_mode: bool = ragged_mode
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting Awkward operations.

        Returns:
            str: The prefix 'ak'.
        """
        return "ak"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helper functions.
        """
        res: list[str] = []
        return res

    def get_ops_map(self, kwargs: dict[str, object]) -> dict[str, str]:
        """Retrieve operation formatting templates for Awkward operations.

        Args:
            kwargs (dict[str, object]): Keyword arguments passed to the visitor.

        Returns:
            dict[str, str]: Map from op_type to formatting template.
        """
        del kwargs
        return {
            # Jagged reductions
            "Sum": "ak.sum({0})",
            "Mean": "ak.mean({0})",
            "Min": "ak.min({0})",
            "Max": "ak.max({0})",
            "Prod": "ak.prod({0})",
            "All": "ak.all({0})",
            "Any": "ak.any({0})",
            "Std": "ak.std({0})",
            "Var": "ak.var({0})",
            "Count": "ak.count({0})",
            "Ptp": "ak.ptp({0})",
            "ArgMin": "ak.argmin({0})",
            "ArgMax": "ak.argmax({0})",
            # Variable-length flattening and structure
            "Flatten": "ak.flatten({0})",
            "Unflatten": "ak.unflatten({0}, {1})",
            "Num": "ak.num({0})",
            "PadNone": "ak.pad_none({0}, {1})",
            "FillNone": "ak.fill_none({0}, {1})",
            "DropNone": "ak.drop_none({0})",
            "Cartesian": "ak.cartesian({0})",
            "Combinations": "ak.combinations({0}, {1})",
            "Concatenate": "ak.concatenate({0})",
            "Where": "ak.where({0}, {1}, {2})",
            # Nested record transformations
            "Zip": "ak.zip({0})",
            "Unzip": "ak.unzip({0})",
            "WithField": "ak.with_field({0}, {1}, {2})",
            "Fields": "ak.fields({0})",
            "ToRegular": "ak.to_regular({0})",
            "FromRegular": "ak.from_regular({0})",
            # Elementwise operations
            "Add": "({0} + {1})",
            "Sub": "({0} - {1})",
            "Subtract": "({0} - {1})",
            "Mul": "({0} * {1})",
            "Multiply": "({0} * {1})",
            "Div": "({0} / {1})",
            "Divide": "({0} / {1})",
            "TrueDivide": "({0} / {1})",
            "Pow": "({0} ** {1})",
            "Power": "({0} ** {1})",
            "Neg": "(-{0})",
            "Abs": "abs({0})",
        }

    def generate(self) -> str:
        """Generate the complete Awkward script.

        Returns:
            str: Generated Python script using Awkward.
        """
        self.code = [self.header]
        self.add_line("import awkward as ak")
        self.add_line("import " + "numpy as np")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes emitting Awkward operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into an executable callable supporting ragged Awkward arrays.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Optional compilation options.

        Returns:
            object: Execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def aot_awkward_runner(*w_args: object, **w_kw: object) -> object:
            """Execute graph with Awkward arrays.

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

        return aot_awkward_runner
