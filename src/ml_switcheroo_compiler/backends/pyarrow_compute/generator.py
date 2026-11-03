"""Apache Arrow Compute code generator targeting zero-copy columnar execution."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor
from ml_switcheroo_compiler.backends.pyarrow_compute.eager import execute_op as _execute_op
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    array as _array,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    arrow_tensor_to_tensor as _arrow_tensor_to_tensor,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    asarray as _asarray,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    check_parity as _check_parity,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    chunked_array as _chunked_array,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    empty as _empty,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    from_arrow as _from_arrow,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    is_arrow_object as _is_arrow_object,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    item as _item,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    ones as _ones,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    record_batch as _record_batch,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    record_batch_to_tensor as _record_batch_to_tensor,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    slice_record_batch as _slice_record_batch,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    table as _table,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    tensor as _tensor,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    tensor_to_arrow_tensor as _tensor_to_arrow_tensor,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    tensor_to_record_batch as _tensor_to_record_batch,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    to_arrow as _to_arrow,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    to_table as _to_table,
)
from ml_switcheroo_compiler.backends.pyarrow_compute.types import (
    zeros as _zeros,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _unwrap_input_val(val: EagerValue) -> EagerValue:
    """Unwrap Arrow object or Tensor to underlying raw array or scalar.

    Args:
        val (EagerValue): Input value.

    Returns:
        EagerValue: Unwrapped raw data.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor

    if _is_arrow_object(val):
        converted = _from_arrow(val)
        return converted.data if isinstance(converted, Tensor) else converted
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

    def get_ops_map(self, kwargs: dict[str, EagerValue] | None = None) -> dict[str, str]:
        """Retrieve operation formatting templates for PyArrow Compute operations.

        Args:
            kwargs (dict[str, EagerValue] | None): Keyword arguments passed to the visitor.

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
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Fallback for generic nodes emitting PyArrow Compute operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an executable callable for Arrow columnar execution.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (EagerValue): Optional compilation options.

        Returns:
            EagerValue: Execution callable.
        """
        from ml_switcheroo_compiler.backends.pyarrow_compute.types import is_arrow_object, to_arrow
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        zero_copy = self.zero_copy

        def aot_arrow_runner(*w_args: EagerValue, **w_kw: EagerValue) -> EagerValue:
            """Execute graph with Arrow columnar structures.

            Args:
                *w_args (EagerValue): Input tensor values.
                **w_kw (EagerValue): Keyword inputs.

            Returns:
                EagerValue: Output results.
            """
            input_nodes = [n for n in graph.nodes.values() if n.op_type == "Input"]
            inputs: dict[str, EagerValue] = {inp_node.id: _unwrap_input_val(w_args[i]) for i, inp_node in enumerate(input_nodes) if i < len(w_args)}
            for k, val in w_kw.items():
                inputs[k] = _unwrap_input_val(val)

            evaluated = evaluate_graph(graph, inputs=inputs)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    res = evaluated.get(graph.outputs[0])
                    if zero_copy and is_arrow_object(w_args[0] if w_args else None):
                        return to_arrow(res, target_type="tensor")
                    return res
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return as_tensor(evaluated)

        return aot_arrow_runner

    @staticmethod
    def zeros(shape: tuple[int, ...]) -> EagerValue:
        """Create a zero-filled Arrow array.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            EagerValue: Zero-filled array.
        """
        return _zeros(shape)

    @staticmethod
    def ones(shape: tuple[int, ...]) -> EagerValue:
        """Create a one-filled Arrow array.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            EagerValue: One-filled array.
        """
        return _ones(shape)

    @staticmethod
    def empty(shape: tuple[int, ...]) -> EagerValue:
        """Create a null-filled Arrow array.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            EagerValue: Null-filled array.
        """
        return _empty(shape)

    @staticmethod
    def array(data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
        """Create a columnar Arrow array from data.

        Args:
            data (EagerValue): Input data.
            dtype (EagerValue | None): Optional dtype string.

        Returns:
            EagerValue: Arrow array.
        """
        dt_str = str(dtype) if dtype is not None else None
        return _array(data, dtype=dt_str)

    @staticmethod
    def asarray(data: EagerValue) -> EagerValue:
        """Convert input to an Arrow columnar array.

        Args:
            data (EagerValue): Input data.

        Returns:
            EagerValue: Arrow array.
        """
        return _asarray(data)

    @staticmethod
    def chunked_array(chunks: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
        """Construct a columnar ChunkedArray.

        Args:
            chunks (EagerValue): Sequence of array chunks.
            dtype (EagerValue | None): Optional Arrow type.

        Returns:
            EagerValue: ChunkedArray.
        """
        return _chunked_array(chunks, dtype=dtype)

    @staticmethod
    def record_batch(
        data: EagerValue,
        names: Sequence[str] | None = None,
        schema: EagerValue | None = None,
    ) -> EagerValue:
        """Create a columnar RecordBatch.

        Args:
            data (EagerValue): Column arrays.
            names (Sequence[str] | None): Field names.
            schema (EagerValue | None): Optional schema.

        Returns:
            EagerValue: RecordBatch.
        """
        return _record_batch(data, names=names, schema=schema)

    @staticmethod
    def slice_record_batch(
        batch: EagerValue,
        offset: int = 0,
        length: int | None = None,
    ) -> EagerValue:
        """Perform zero-copy slicing of a RecordBatch.

        Args:
            batch (EagerValue): Target RecordBatch.
            offset (int): Starting index.
            length (int | None): Slice length.

        Returns:
            EagerValue: Sliced RecordBatch.
        """
        return _slice_record_batch(batch, offset=offset, length=length)

    @staticmethod
    def to_table(
        data: EagerValue,
        names: Sequence[str] | None = None,
        schema: EagerValue | None = None,
    ) -> EagerValue:
        """Convert data to an Arrow Table.

        Args:
            data (EagerValue): Input data.
            names (Sequence[str] | None): Optional column names.
            schema (EagerValue | None): Optional schema.

        Returns:
            EagerValue: Arrow Table.
        """
        return _to_table(data, names=names, schema=schema)

    @staticmethod
    def table(
        data: EagerValue,
        names: Sequence[str] | None = None,
        schema: EagerValue | None = None,
    ) -> EagerValue:
        """Construct an Arrow Table.

        Args:
            data (EagerValue): Input data.
            names (Sequence[str] | None): Optional column names.
            schema (EagerValue | None): Optional schema.

        Returns:
            EagerValue: Arrow Table.
        """
        return _table(data, names=names, schema=schema)

    @staticmethod
    def tensor(data: EagerValue) -> EagerValue:
        """Construct a pyarrow.Tensor.

        Args:
            data (EagerValue): Input data.

        Returns:
            EagerValue: pyarrow.Tensor.
        """
        return _tensor(data)

    @staticmethod
    def check_parity(
        lhs: EagerValue,
        rhs: EagerValue,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Check numerical parity between two Arrow structures.

        Args:
            lhs (EagerValue): First operand.
            rhs (EagerValue): Second operand.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.

        Returns:
            bool: True if matching.
        """
        return _check_parity(lhs, rhs, rtol=rtol, atol=atol)

    @staticmethod
    def item(data: EagerValue) -> float:
        """Extract scalar item from Arrow data.

        Args:
            data (EagerValue): Target data.

        Returns:
            float: Extracted scalar.
        """
        return _item(data)

    @staticmethod
    def execute_op(op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Execute operation eagerly on PyArrow Compute backend.

        Args:
            op_type (str): Operation type.
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            EagerValue: Computed result.
        """
        return _execute_op(op_type, *args, **kwargs)

    @staticmethod
    def tensor_to_arrow_tensor(tensor_val: EagerValue) -> EagerValue:
        """Convert Tensor to pyarrow.Tensor.

        Args:
            tensor_val (EagerValue): Input Tensor.

        Returns:
            EagerValue: pyarrow.Tensor.
        """
        return _tensor_to_arrow_tensor(tensor_val)

    @staticmethod
    def arrow_tensor_to_tensor(arrow_tensor: EagerValue) -> EagerValue:
        """Convert pyarrow.Tensor to Unified IR Tensor.

        Args:
            arrow_tensor (EagerValue): Input pyarrow.Tensor.

        Returns:
            EagerValue: Unified IR Tensor.
        """
        return _arrow_tensor_to_tensor(arrow_tensor)

    @staticmethod
    def tensor_to_record_batch(
        tensor_val: EagerValue,
        names: Sequence[str] | None = None,
    ) -> EagerValue:
        """Convert Tensor to Arrow RecordBatch.

        Args:
            tensor_val (EagerValue): Input Tensor.
            names (Sequence[str] | None): Optional column names.

        Returns:
            EagerValue: Arrow RecordBatch.
        """
        return _tensor_to_record_batch(tensor_val, names=names)

    @staticmethod
    def record_batch_to_tensor(
        batch: EagerValue,
        columns: Sequence[str] | None = None,
        dtype: str | None = None,
    ) -> EagerValue:
        """Convert Arrow RecordBatch to Unified IR Tensor.

        Args:
            batch (EagerValue): Input RecordBatch.
            columns (Sequence[str] | None): Optional column subset.
            dtype (str | None): Optional target dtype.

        Returns:
            EagerValue: Unified IR Tensor.
        """
        return _record_batch_to_tensor(batch, columns=columns, dtype=dtype)

    @staticmethod
    def is_arrow_object(obj: EagerValue) -> bool:
        """Check if object is a PyArrow construct.

        Args:
            obj (EagerValue): Candidate object.

        Returns:
            bool: True if PyArrow construct.
        """
        return _is_arrow_object(obj)

    @staticmethod
    def to_arrow(data: EagerValue, target_type: str = "tensor") -> EagerValue:
        """Convert Tensor to specified Arrow type.

        Args:
            data (EagerValue): Input Tensor.
            target_type (str): Target Arrow type.

        Returns:
            EagerValue: Converted Arrow object.
        """
        return _to_arrow(data, target_type=target_type)

    @staticmethod
    def from_arrow(data: EagerValue) -> EagerValue:
        """Convert Arrow structure to Unified IR Tensor.

        Args:
            data (EagerValue): Input Arrow object.

        Returns:
            EagerValue: Unified IR Tensor.
        """
        return _from_arrow(data)
