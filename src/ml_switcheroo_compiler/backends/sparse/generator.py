"""Sparse Coordinate (COO) code generator and compilation pipeline."""

from __future__ import annotations

import logging

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, StaticEagerBackendAdapter
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.sparse.eager import execute_op as _sparse_execute_op
from ml_switcheroo_compiler.backends.sparse.types import (
    COOTensor,
)
from ml_switcheroo_compiler.backends.sparse.types import (
    array as _sparse_array,
)
from ml_switcheroo_compiler.backends.sparse.types import (
    asarray as _sparse_asarray,
)
from ml_switcheroo_compiler.backends.sparse.types import (
    item as _sparse_item,
)
from ml_switcheroo_compiler.backends.sparse.types import (
    zeros as _sparse_zeros,
)
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable, extract_data


@register_backend("sparse_coo")
@register_backend("sparse")
class SparseGenerator(PythonStringGenerator):
    """Generate executable Python code using dedicated Sparse COO tensor kernels."""

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> COOTensor:
        """Create a Sparse COO tensor of zeros with the given shape.

        Args:
            shape (tuple[int, ...]): Target tensor shape.

        Returns:
            COOTensor: Zero-initialized COO tensor.
        """
        return _sparse_zeros(cls, shape)

    @classmethod
    def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> COOTensor:
        """Create a Sparse COO tensor from input data.

        Args:
            data (BackendArg): Input array or sequence.
            dtype (BackendArg | None): Optional target dtype.

        Returns:
            COOTensor: Constructed COO tensor.
        """
        return _sparse_array(cls, data, dtype=str(dtype) if dtype is not None else None)

    @classmethod
    def asarray(cls, data: BackendArg) -> COOTensor:
        """Convert input data to a Sparse COO tensor.

        Args:
            data (BackendArg): Input array or sequence.

        Returns:
            COOTensor: Converted COO tensor.
        """
        return _sparse_asarray(cls, data)

    @classmethod
    def item(cls, data: BackendArg) -> float:
        """Extract scalar float value from a Sparse COO tensor.

        Args:
            data (BackendArg): Input 0D/1-element tensor.

        Returns:
            float: Extracted scalar value.
        """
        return _sparse_item(cls, data)

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an operation eagerly on the Sparse COO backend.

        Args:
            op_type (str): Canonical operation name.
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Evaluated operation result.
        """
        return _sparse_execute_op(cls, op_type, *args, **kwargs)

    def __init__(self, graph: IRGraph, memory_efficient: bool = True) -> None:
        """Initialize the Sparse COO code generator.

        Args:
            graph (IRGraph): The computation graph to emit code for.
            memory_efficient (bool): Whether to generate memory-efficient sparse calls.
        """
        super().__init__(graph)
        self.memory_efficient: bool = memory_efficient
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the module prefix string used for sparse kernel operations.

        Returns:
            str: The prefix 'sp_kernels'.
        """
        return "sp_kernels"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helpers.
        """
        res: list[str] = []
        return res

    def get_ops_map(self, kwargs: dict[str, BackendArg] | None = None) -> dict[str, str]:
        """Retrieve operation formatting templates for Sparse COO kernels.

        Args:
            kwargs (dict[str, BackendArg] | None): Keyword arguments passed to the visitor.

        Returns:
            dict[str, str]: Map from op_type to formatting template.
        """
        del kwargs
        return {
            "coo_fromdense": "sp_kernels.coo_fromdense({0})",
            "coo_todense": "sp_kernels.coo_todense({0})",
            "coo_matmat": "sp_kernels.coo_matmat({0}, {1})",
            "coo_matvec": "sp_kernels.coo_matvec({0}, {1})",
            "Add": "sp_kernels.coo_add({0}, {1})",
            "Sub": "sp_kernels.coo_sub({0}, {1})",
            "Mul": "sp_kernels.coo_mul({0}, {1})",
            "Neg": "sp_kernels.coo_neg({0})",
            "Abs": "sp_kernels.coo_abs({0})",
            "Transpose": "sp_kernels.coo_transpose({0})",
            "Sum": "sp_kernels.coo_sum({0})",
            "Mean": "sp_kernels.coo_mean({0})",
            "Relu": "sp_kernels.coo_relu({0})",
            "Reshape": "sp_kernels.coo_reshape({0}, {shape})",
            "Dot": "sp_kernels.coo_dot({0}, {1})",
            "MatMul": "sp_kernels.coo_matmat({0}, {1})",
            "spmm": "sp_kernels.spmm({0}, {1})",
            "spgemm": "sp_kernels.spgemm({0}, {1})",
            "dense_spmm": "sp_kernels.dense_spmm({0}, {1})",
            "csr_add": "sp_kernels.csr_add({0}, {1})",
            "csc_add": "sp_kernels.csc_add({0}, {1})",
            "csr_sub": "sp_kernels.csr_sub({0}, {1})",
            "csc_sub": "sp_kernels.csc_sub({0}, {1})",
            "sparse_mask": "sp_kernels.sparse_mask({0}, {1})",
            "sparse_conv2d_mask": "sp_kernels.sparse_conv2d_mask({0}, {1}, {2})",
            "sparse_softmax": "sp_kernels.sparse_softmax({0})",
            "graph_norm_adjacency": "sp_kernels.graph_norm_adjacency({0})",
            "gnn_spmm_attention": "sp_kernels.gnn_spmm_attention({0}, {1})",
            "sparse_dropout": "sp_kernels.sparse_dropout({0})",
        }

    def generate(self) -> str:
        """Generate complete Python source script executing sparse operations.

        Returns:
            str: Generated Python script string.
        """
        self.code = [self.header]
        self.add_line("from ml_switcheroo_compiler.backends.sparse import kernels as sp_kernels")
        self.add_line("from ml_switcheroo_compiler.backends.sparse.types import COOTensor, CSCTensor, CSRTensor")
        if self.memory_efficient:
            self.add_line("# Memory-efficient sparse execution active")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def compile_fn(self) -> TypedVariadicCallable[BackendArg]:
        """Compile the generated Python code into an executable callable.

        Returns:
            TypedVariadicCallable[BackendArg]: Compiled function.

        Raises:
            RuntimeError: If evaluation function was not defined in generated script.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code

        source: str = self.generate()
        namespace = load_generated_code(source)
        func: BackendArg | None = namespace.get(self._func_name)
        if not callable(func):
            msg = f"Generated code did not produce callable '{self._func_name}'."
            raise RuntimeError(msg)
        from ml_switcheroo_compiler.backends.eager_registry import require_callable

        bound_fn = require_callable(func)

        def _typed_runner(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Invoke compiled sparse evaluation function.

            Args:
                *args (BackendArg): Positional arguments.
                **kwargs (BackendArg): Keyword arguments.

            Returns:
                BackendArg: Evaluated sparse output.
            """
            return bound_fn(*args, **kwargs)

        return _typed_runner

    def visit_Spmm(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse-dense matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.spmm({', '.join(input_vars)})"

    def visit_Spgemm(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse-sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.spgemm({', '.join(input_vars)})"

    def visit_DenseSpmm(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit dense-sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.dense_spmm({', '.join(input_vars)})"

    def visit_SparseMask(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse masking operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_mask({', '.join(input_vars)})"

    def visit_SparseSoftmax(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse softmax kernel operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_softmax({', '.join(input_vars)})"

    def visit_GraphNormAdjacency(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit graph normalized adjacency kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.graph_norm_adjacency({', '.join(input_vars)})"

    def visit_GnnSpmmAttention(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit GNN attention aggregation kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.gnn_spmm_attention({', '.join(input_vars)})"

    def visit_SparseDropout(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse dropout regularizer.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_dropout({', '.join(input_vars)})"

    def visit_CsrAdd(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit CSR addition kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csr_add({', '.join(input_vars)})"

    def visit_CscAdd(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit CSC addition kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csc_add({', '.join(input_vars)})"

    def visit_CsrSub(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit CSR subtraction kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csr_sub({', '.join(input_vars)})"

    def visit_CscSub(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit CSC subtraction kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csc_sub({', '.join(input_vars)})"

    def visit_Add(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse COO addition.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_add({', '.join(input_vars)})"

    def visit_Sub(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse COO subtraction.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_sub({', '.join(input_vars)})"

    def visit_Mul(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse COO elementwise multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_mul({', '.join(input_vars)})"

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (BackendArg): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_matmat({', '.join(input_vars)})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Handle fallback emission for nodes not explicitly specialized.

        Args:
            node (IRNode): Computation graph node.
            input_vars (list[str]): Input variable names.
            **kwargs (BackendArg): Node attributes and keyword options.

        Returns:
            str: Generated code expression string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into ahead-of-time Sparse COO executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (BackendArg): Compiler options.

        Returns:
            BackendArg: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact

        source_code = self.generate()
        compiled_fn = self.compile_fn()

        def runner(*args: BackendArg, **runner_kwargs: BackendArg) -> BackendArg:
            """Execute compiled Sparse artifact or fallback graph evaluation.

            Args:
                *args (BackendArg): Input tensor or sparse matrix arguments.
                **runner_kwargs (BackendArg): Optional execution options.

            Returns:
                BackendArg: Output tensor or tuple of output tensors.
            """
            del runner_kwargs
            if compiled_fn is not None:
                try:
                    args_list = [extract_data(a) for a in args]
                    return compiled_fn(args_list)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            feed_dict: dict[str, BackendArg] = {}
            for idx, arg in enumerate(args):
                if idx < len(graph.inputs):
                    feed_dict[graph.inputs[idx]] = extract_data(arg)
            active_be = get_active_backend()
            eval_be = BackendRegistry.get("pure_python") if active_be.__name__ == "SparseCodeGenerator" else active_be
            res_dict = evaluate_graph(graph, feed_dict, backend=eval_be)
            if not graph.outputs:
                return res_dict
            if len(graph.outputs) == 1:
                return res_dict.get(graph.outputs[0])
            return tuple(res_dict.get(out) for out in graph.outputs)

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=source_code.encode("utf-8"),
            source_code=source_code,
            metadata={"backend": "sparse_coo", "options": kwargs},
        )

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for Sparse COO.

        Returns:
            EagerBackendAdapter: Concrete Sparse COO eager backend adapter.
        """
        import sparse

        from . import kernels as sp_k

        return StaticEagerBackendAdapter(
            backend_name="sparse",
            ops={
                "abs": sp_k.coo_abs,
                "add": sp_k.coo_add,
                "sub": sp_k.coo_sub,
                "subtract": sp_k.coo_sub,
                "mul": sp_k.coo_mul,
                "multiply": sp_k.coo_mul,
                "neg": sp_k.coo_neg,
                "relu": sp_k.coo_relu,
                "mean": sp_k.coo_mean,
                "dot": sp_k.coo_dot,
                "matmul": sp_k.coo_dot,
                "sum": sp_k.coo_sum,
                "transpose": sp_k.coo_transpose,
                "reshape": sp_k.coo_reshape,
                "todense": sp_k.coo_todense,
                "fromdense": sp_k.coo_fromdense,
                "zeros": cls.zeros,
                "array": cls.array,
                "asarray": cls.asarray,
            },
            module=sparse,
        )
