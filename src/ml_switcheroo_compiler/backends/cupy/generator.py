"""CuPy code generator and eager execution backend."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.ir.core import IRGraph

try:
    import cupy as cp
except ImportError:
    cp = None

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("cupy")
class CupyGenerator(PythonStringGenerator):
    """Generate CuPy python code from IR."""

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): The graph parameter.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "cp"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions.

        Returns:
            list[str]: Result.
        """
        res: list[str] = []
        return res

    _import_header = "import cupy as cp"
    _func_name = "evaluate"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Fallback for generic nodes.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): The input_vars parameter.
            **kwargs: Extra attributes.

        Returns:
            str: Generated code.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an AOT execution callable with CUDA graph capture or compiled kernel.

        Args:
            graph (IRGraph): Target computation graph.
            **kwargs (EagerValue): Optional compiler options ('sample_inputs').

        Returns:
            EagerValue: Callable execution wrapper.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def forward_fn(*fn_args: EagerValue) -> EagerValue:
            """Evaluate the graph with CuPy arrays.

            Args:
                *fn_args (EagerValue): Input tensor values.

            Returns:
                EagerValue: Output tensor or collection of tensors.
            """
            input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
            inputs: dict[str, EagerValue] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(fn_args):
                    arg_val = fn_args[i]
                    inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
            evaluated = evaluate_graph(graph, inputs=inputs)
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        try:
            import cupy as cp

            sample_inputs = kwargs.get("sample_inputs")
            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)) and hasattr(cp.cuda, "Graph"):
                cp_samples = [cp.asarray(x) for x in sample_inputs]
                forward_fn(*cp_samples)
                stream = cp.cuda.Stream()
                with stream:
                    graph_record = cp.cuda.Graph()
                    graph_record.begin_capture()
                    out_static = forward_fn(*cp_samples)
                    graph_record.end_capture()
                    instance = graph_record.instantiate()

                def aot_cupy_graph_runner(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
                    """Execute captured CuPy CUDA graph.

                    Args:
                        *w_args (EagerValue): Input tensors.
                        **w_kwargs (EagerValue): Keyword arguments.

                    Returns:
                        EagerValue: Computed outputs.
                    """
                    instance.launch(stream)
                    stream.synchronize()
                    return out_static

                return aot_cupy_graph_runner

            def aot_cupy_runner(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
                """Execute compiled CuPy graph callable.

                Args:
                    *w_args (EagerValue): Input tensors.
                    **w_kwargs (EagerValue): Keyword arguments.

                Returns:
                    EagerValue: Computed outputs.
                """
                cp_args = [cp.asarray(a.data if isinstance(a, Tensor) else a) for a in w_args]
                return forward_fn(*cp_args)

            return aot_cupy_runner
        except Exception:
            return forward_fn


CuPyGenerator = CupyGenerator
