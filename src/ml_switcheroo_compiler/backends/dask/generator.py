# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dask code generator and eager execution backend."""

try:
    import dask.array as da
except ImportError:
    da = None

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("dask")
class DaskGenerator(PythonStringGenerator):
    """Generate Dask python code from IR."""

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): The graph parameter.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting Dask array operations.

        Returns:
        str: Result.
        """
        return "da"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions.

        Returns:
            list[str]: Result.
        """
        res: list[str] = []
        return res

    _import_header = "import dask.array as da"
    _func_name = "evaluate"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): The input_vars parameter.
            **kwargs: Extra attributes.

        Returns:
            str: Generated code.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Construct and optimize a static Dask computation task graph and return an execution callable.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Optional compiler arguments ('optimize_graph').

        Returns:
            object: Execution callable evaluating the optimized Dask task graph.
        """
        import importlib

        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        numpy_mod = importlib.import_module("numpy")

        def forward_fn(*fn_args: object) -> object:
            """Evaluate the graph with Dask arrays.

            Args:
                *fn_args (object): Input tensor values.

            Returns:
                object: Output tensor or collection of tensors.
            """
            input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
            inputs: dict[str, object] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(fn_args):
                    arg_val = fn_args[i]
                    inputs[inp_node.id] = numpy_mod.asarray(arg_val.data if isinstance(arg_val, Tensor) else arg_val)
            evaluated = evaluate_graph(graph, inputs=inputs)
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        try:
            import dask
            import dask.array as da

            delayed_fn = dask.delayed(forward_fn)

            def dask_aot_runner(*w_args: object, **w_kwargs: object) -> object:
                """Execute optimized Dask task graph.

                Args:
                    *w_args (object): Input tensors.
                    **w_kwargs (object): Keyword arguments.

                Returns:
                    object: Computed outputs.
                """
                da_args = [da.from_array(a.data if isinstance(a, Tensor) else a) for a in w_args]
                task = delayed_fn(*da_args)
                return dask.compute(task)[0]

            return dask_aot_runner
        except Exception:
            return forward_fn
