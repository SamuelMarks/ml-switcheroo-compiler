"""Apache Arrow Compute code generator targeting zero-copy columnar execution."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _unwrap_input_val(val: object) -> object:
    """Unwrap Arrow object or Tensor to underlying raw array or scalar.

    Args:
        val (object): Input value.

    Returns:
        object: Unwrapped raw data.
    """
    from ml_switcheroo_compiler.backends.pyarrow_compute.types import from_arrow, is_arrow_object
    from ml_switcheroo_compiler.core.tensor import Tensor

    if is_arrow_object(val):
        converted = from_arrow(val)
        return getattr(converted, "data", converted)
    if isinstance(val, Tensor):
        return val.data
    return val


@register_backend("pyarrow_compute")
class PyArrowComputeGenerator(PythonStringGenerator):
    """Generate Python/PyArrow code operating directly on columnar memory buffers."""

    def __init__(
        self,
        graph: IRGraph,
        zero_copy: bool = True,
    ) -> None:
        """Initialize the PyArrow Compute code generator.

        Args:
            graph (IRGraph): The computation graph to compile.
            zero_copy (bool): Whether to enforce zero-copy columnar buffer views.
        """
        super().__init__(graph)
        self.zero_copy: bool = zero_copy
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting Arrow Compute operations.

        Returns:
            str: The prefix 'pc'.
        """
        return "pc"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helper functions.
        """
        res: list[str] = []
        return res

    def get_ops_map(self, kwargs: dict[str, object]) -> dict[str, str]:
        """Retrieve operation formatting templates for PyArrow Compute operations.

        Args:
            kwargs (dict[str, object]): Keyword arguments passed to the visitor.

        Returns:
            dict[str, str]: Map from op_type to formatting template.
        """
        del kwargs
        return {
            "Add": "pc.add({0}, {1})",
            "Sub": "pc.subtract({0}, {1})",
            "Subtract": "pc.subtract({0}, {1})",
            "Mul": "pc.multiply({0}, {1})",
            "Multiply": "pc.multiply({0}, {1})",
            "Div": "pc.divide({0}, {1})",
            "Divide": "pc.divide({0}, {1})",
            "TrueDivide": "pc.divide({0}, {1})",
            "Pow": "pc.power({0}, {1})",
            "Power": "pc.power({0}, {1})",
            "Neg": "pc.negate({0})",
            "Negative": "pc.negate({0})",
            "Abs": "pc.abs({0})",
            "Absolute": "pc.abs({0})",
            "Sign": "pc.sign({0})",
            "Sqrt": "pc.sqrt({0})",
            "Exp": "pc.exp({0})",
            "Log": "pc.ln({0})",
            "Ln": "pc.ln({0})",
            "Log10": "pc.log10({0})",
            "Log2": "pc.log2({0})",
            "Log1p": "pc.log1p({0})",
            "Expm1": "pc.expm1({0})",
            "Floor": "pc.floor({0})",
            "Ceil": "pc.ceil({0})",
            "Round": "pc.round({0})",
            "Trunc": "pc.trunc({0})",
            "Sin": "pc.sin({0})",
            "Cos": "pc.cos({0})",
            "Tan": "pc.tan({0})",
            "Asin": "pc.asin({0})",
            "Acos": "pc.acos({0})",
            "Atan": "pc.atan({0})",
            "Atan2": "pc.atan2({0}, {1})",
            "Equal": "pc.equal({0}, {1})",
            "NotEqual": "pc.not_equal({0}, {1})",
            "Greater": "pc.greater({0}, {1})",
            "GreaterEqual": "pc.greater_equal({0}, {1})",
            "Less": "pc.less({0}, {1})",
            "LessEqual": "pc.less_equal({0}, {1})",
            "And": "pc.and_({0}, {1})",
            "Or": "pc.or_({0}, {1})",
            "Xor": "pc.xor({0}, {1})",
            "Not": "pc.invert({0})",
            "Minimum": "pc.min_element_wise({0}, {1})",
            "Maximum": "pc.max_element_wise({0}, {1})",
            "Sum": "pc.sum({0})",
            "Mean": "pc.mean({0})",
            "Min": "pc.min({0})",
            "Max": "pc.max({0})",
            "All": "pc.all({0})",
            "Any": "pc.any({0})",
            "Std": "pc.stddev({0})",
            "Var": "pc.variance({0})",
            "Count": "pc.count({0})",
            "Prod": "pc.product({0})",
            "Product": "pc.product({0})",
            "Mode": "pc.mode({0})",
            "Quantile": "pc.quantile({0})",
            "Median": "pc.approximate_median({0})",
            "CumSum": "pc.cumulative_sum({0})",
            "CumulativeSum": "pc.cumulative_sum({0})",
            "CumProd": "pc.cumulative_prod({0})",
            "CumulativeProd": "pc.cumulative_prod({0})",
            "CumMax": "pc.cumulative_max({0})",
            "CumMin": "pc.cumulative_min({0})",
            "Where": "pc.if_else({0}, {1}, {2})",
            "Cast": "pc.cast({0}, {1})",
            "Filter": "pc.filter({0}, {1})",
            "Take": "pc.take({0}, {1})",
            "Gather": "pc.take({0}, {1})",
            "DropNull": "pc.drop_null({0})",
            "FillNull": "pc.fill_null({0}, {1})",
            "ReplaceWithMask": "pc.replace_with_mask({0}, {1}, {2})",
            "Unique": "pc.unique({0})",
            "ValueCounts": "pc.value_counts({0})",
            "SortIndices": "pc.sort_indices({0})",
            "Rank": "pc.rank({0})",
            "DictionaryEncode": "pc.dictionary_encode({0})",
            "IndicesNonzero": "pc.indices_nonzero({0})",
            "ListFlatten": "pc.list_flatten({0})",
            "Flatten": "pc.list_flatten({0})",
            "ListSlice": "pc.list_slice({0}, {1}, {2})",
            "ListElement": "pc.list_element({0}, {1})",
            "MakeStruct": "pc.make_struct({0})",
            "StructField": "pc.struct_field({0}, {1})",
            "Lower": "pc.utf8_lower({0})",
            "Upper": "pc.utf8_upper({0})",
            "StringLength": "pc.utf8_length({0})",
            "ReplaceSubstring": "pc.replace_substring({0}, {1}, {2})",
            "MatchSubstring": "pc.match_substring({0}, {1})",
            "Cbrt": "pc.cbrt({0})",
            "ShiftLeft": "pc.shift_left({0}, {1})",
            "ShiftRight": "pc.shift_right({0}, {1})",
            "BitwiseAnd": "pc.bit_wise_and({0}, {1})",
            "BitwiseOr": "pc.bit_wise_or({0}, {1})",
            "BitwiseXor": "pc.bit_wise_xor({0}, {1})",
            "BitwiseNot": "pc.bit_wise_not({0})",
            "IsNull": "pc.is_null({0})",
            "IsValid": "pc.is_valid({0})",
            "IsNan": "pc.is_nan({0})",
            "IsInf": "pc.is_inf({0})",
            "IsFinite": "pc.is_finite({0})",
        }

    def generate(self) -> str:
        """Generate the complete Arrow Compute script.

        Returns:
            str: Generated Python script using PyArrow Compute.
        """
        self.code = [self.header]
        self.add_line("import pyarrow.compute as pc")
        self.add_line("import pyarrow as pa")
        self.add_line("import " + "numpy as np")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes emitting PyArrow Compute operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into an executable callable for Arrow columnar execution.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Optional compilation options.

        Returns:
            object: Execution callable.
        """
        from ml_switcheroo_compiler.backends.pyarrow_compute.types import is_arrow_object, to_arrow
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        zero_copy = self.zero_copy

        def aot_arrow_runner(*w_args: object, **w_kw: object) -> object:
            """Execute graph with Arrow columnar structures.

            Args:
                *w_args (object): Input tensor values.
                **w_kw (object): Keyword inputs.

            Returns:
                object: Output results.
            """
            input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
            inputs: dict[str, object] = {inp_node.id: _unwrap_input_val(w_args[i]) for i, inp_node in enumerate(input_nodes) if i < len(w_args)}
            for k, val in w_kw.items():
                inputs[k] = _unwrap_input_val(val)

            evaluated = evaluate_graph(graph, inputs=inputs)
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    res = evaluated.get(graph.outputs[0])
                    if zero_copy and is_arrow_object(w_args[0] if w_args else None):
                        return to_arrow(res, target_type="tensor")
                    return res
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        return aot_arrow_runner
