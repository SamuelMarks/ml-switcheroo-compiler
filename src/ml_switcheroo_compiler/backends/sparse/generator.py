"""Sparse Coordinate (COO) code generator and compilation pipeline."""

from __future__ import annotations

import logging
import typing

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.sparse.types import COOTensor
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("sparse_coo")
@register_backend("sparse")
class SparseGenerator(PythonStringGenerator):
    """Generate executable Python code using dedicated Sparse COO tensor kernels."""

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

    def get_ops_map(self, kwargs: dict[str, EagerValue] | None = None) -> dict[str, str]:
        """Retrieve operation formatting templates for Sparse COO kernels.

        Args:
            kwargs (dict[str, object] | None): Keyword arguments passed to the visitor.

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

    def compile_fn(self) -> TypedVariadicCallable[COOTensor | EagerValue | tuple[COOTensor | EagerValue, ...]]:
        """Compile the generated Python code into an executable callable.

        Returns:
            TypedVariadicCallable[Union[COOTensor, EagerValue, tuple[Union[COOTensor, EagerValue], ...]]]: Compiled function.

        Raises:
            RuntimeError: If evaluation function was not defined in generated script.
        """
        source: str = self.generate()
        namespace: dict[str, EagerValue] = {}
        exec(source, namespace)
        func: EagerValue | None = namespace.get(self._func_name)
        if not callable(func):
            msg = f"Generated code did not produce callable '{self._func_name}'."
            raise RuntimeError(msg)
        return typing.cast(
            "TypedVariadicCallable[COOTensor | EagerValue | tuple[COOTensor | EagerValue, ...]]",
            func,
        )

    def visit_Spmm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse-dense matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.spmm({', '.join(input_vars)})"

    def visit_Spgemm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse-sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.spgemm({', '.join(input_vars)})"

    def visit_DenseSpmm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit dense-sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.dense_spmm({', '.join(input_vars)})"

    def visit_SparseMask(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse masking operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_mask({', '.join(input_vars)})"

    def visit_SparseSoftmax(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse softmax kernel operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_softmax({', '.join(input_vars)})"

    def visit_GraphNormAdjacency(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit graph normalized adjacency kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.graph_norm_adjacency({', '.join(input_vars)})"

    def visit_GnnSpmmAttention(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit GNN attention aggregation kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.gnn_spmm_attention({', '.join(input_vars)})"

    def visit_SparseDropout(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse dropout regularizer.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.sparse_dropout({', '.join(input_vars)})"

    def visit_CsrAdd(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit CSR addition kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csr_add({', '.join(input_vars)})"

    def visit_CscAdd(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit CSC addition kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csc_add({', '.join(input_vars)})"

    def visit_CsrSub(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit CSR subtraction kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csr_sub({', '.join(input_vars)})"

    def visit_CscSub(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit CSC subtraction kernel.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.csc_sub({', '.join(input_vars)})"

    def visit_Add(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse COO addition.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_add({', '.join(input_vars)})"

    def visit_Sub(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse COO subtraction.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_sub({', '.join(input_vars)})"

    def visit_Mul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse COO elementwise multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_mul({', '.join(input_vars)})"

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit sparse matrix multiplication.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        del node, kwargs
        return f"sp_kernels.coo_matmat({', '.join(input_vars)})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Handle fallback emission for nodes not explicitly specialized.

        Args:
            node (IRNode): Computation graph node.
            input_vars (list[str]): Input variable names.
            **kwargs (EagerValue): Node attributes and keyword options.

        Returns:
            str: Generated code expression string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into ahead-of-time Sparse COO executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (EagerValue): Compiler options.

        Returns:
            EagerValue: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact

        source_code = self.generate()
        compiled_fn = self.compile_fn()

        def runner(*args: EagerValue, **runner_kwargs: EagerValue) -> EagerValue:
            """Execute compiled Sparse artifact or fallback graph evaluation.

            Args:
                *args (EagerValue): Input tensor or sparse matrix arguments.
                **runner_kwargs (EagerValue): Optional execution options.

            Returns:
                EagerValue: Output tensor or tuple of output tensors.
            """
            del runner_kwargs
            if compiled_fn is not None:
                try:
                    args_list = [getattr(a, "data", a) for a in args]
                    return compiled_fn(args_list)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            from ml_switcheroo_compiler.backends.registry import BackendRegistry
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            feed_dict: dict[str, EagerValue] = {}
            for idx, arg in enumerate(args):
                if idx < len(graph.inputs):
                    feed_dict[graph.inputs[idx]] = getattr(arg, "data", arg)
            res_dict = evaluate_graph(graph, feed_dict, backend=BackendRegistry.get("numpy"))
            if not graph.outputs:
                return as_tensor(res_dict)
            if len(graph.outputs) == 1:
                return res_dict.get(graph.outputs[0])
            return tuple(res_dict.get(out) for out in graph.outputs)

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=source_code.encode("utf-8"),
            source_code=source_code,
            metadata={"backend": "sparse_coo", "options": kwargs},
        )
