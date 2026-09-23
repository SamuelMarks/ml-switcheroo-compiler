"""Sparse Coordinate (COO) code generator and compilation pipeline."""

from collections.abc import Callable
from typing import Optional, Union

import numpy as np

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.sparse.types import COOTensor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("sparse_coo")
@register_backend("sparse")
class SparseGenerator(PythonStringGenerator):
    """Generate executable Python code using dedicated Sparse COO tensor kernels."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the Sparse COO code generator.

        Args:
            graph (IRGraph): The computation graph to emit code for.
        """
        super().__init__(graph)
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

    def get_ops_map(self, kwargs: dict[str, object]) -> dict[str, str]:
        """Retrieve operation formatting templates for Sparse COO kernels.

        Args:
            kwargs (dict[str, object]): Keyword arguments passed to the visitor.

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
            "sparse_mask": "sp_kernels.sparse_mask({0}, {1})",
            "sparse_conv2d_mask": "sp_kernels.sparse_conv2d_mask({0}, {1}, {2})",
        }

    def generate(self) -> str:
        """Generate complete Python source script executing sparse operations.

        Returns:
            str: Generated Python script string.
        """
        self.code = [self.header]
        self.add_line("import numpy as np")
        self.add_line("from ml_switcheroo_compiler.backends.sparse import kernels as sp_kernels")
        self.add_line("from ml_switcheroo_compiler.backends.sparse.types import COOTensor")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def compile_fn(self) -> Callable[..., Union[COOTensor, np.ndarray, tuple[Union[COOTensor, np.ndarray], ...]]]:
        """Compile the generated Python code into an executable callable.

        Returns:
            Callable[..., Union[COOTensor, np.ndarray, tuple[Union[COOTensor, np.ndarray], ...]]]: Compiled function.

        Raises:
            RuntimeError: If evaluation function was not defined in generated script.
        """
        source: str = self.generate()
        namespace: dict[str, object] = {}
        exec(source, namespace)  # noqa: S102
        func: Optional[object] = namespace.get(self._func_name)
        if not callable(func):
            msg = f"Generated code did not produce callable '{self._func_name}'."
            raise RuntimeError(msg)
        return func

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle fallback emission for nodes not explicitly specialized.

        Args:
            node (IRNode): Computation graph node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Node attributes and keyword options.

        Returns:
            str: Generated code expression string.
        """
        return super().generic_visit(node, input_vars, **kwargs)
