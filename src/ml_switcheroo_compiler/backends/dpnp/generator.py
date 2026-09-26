"""Data Parallel NumPy (dpnp) code generator targeting Intel SYCL hardware."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("dpnp")
class DPNPGenerator(PythonStringGenerator):
    """Generate hardware-accelerated Python/DPNP code targeting Intel SYCL devices."""

    def __init__(
        self,
        graph: IRGraph,
        device: str = "auto",
        sycl_queue: object | None = None,
    ) -> None:
        """Initialize the DPNP code generator.

        Args:
            graph (IRGraph): The computation graph to compile.
            device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
            sycl_queue (object | None): Optional pre-initialized SYCL queue.
        """
        super().__init__(graph)
        self.device: str = device
        self.sycl_queue: object | None = sycl_queue
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting DPNP operations.

        Returns:
            str: The prefix 'dpnp'.
        """
        return "dpnp"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helper functions.
        """
        res: list[str] = []
        return res

    def generate(self) -> str:
        """Generate the complete DPNP script with SYCL queue target.

        Returns:
            str: Generated Python script using DPNP.
        """
        self.code = [self.header]
        self.add_line("import dpnp")
        if self.device != "auto":
            self.add_line(f"# Target SYCL Device: {self.device}")
        if self.sycl_queue is not None:
            self.add_line("# Active SYCL Queue configuration attached")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def visit_Add(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit DPNP addition operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.add({', '.join(input_vars)})"

    def visit_Sub(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit DPNP subtraction operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.subtract({', '.join(input_vars)})"

    def visit_Mul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit DPNP multiplication operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.multiply({', '.join(input_vars)})"

    def visit_Div(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit DPNP division operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.divide({', '.join(input_vars)})"

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit DPNP matrix multiplication operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.matmul({', '.join(input_vars)})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes emitting DPNP operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into an executable callable targeting Intel SYCL.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Optional compilation options.

        Returns:
            object: Execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        generator_cls = self.__class__

        def aot_dpnp_runner(*w_args: object, **w_kw: object) -> object:
            """Execute graph targeting DPNP device.

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
            evaluated = evaluate_graph(graph, inputs=inputs, backend=generator_cls)
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        return aot_dpnp_runner
