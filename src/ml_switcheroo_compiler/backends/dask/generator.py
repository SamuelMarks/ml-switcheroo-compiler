"""Dask code generator and eager execution backend."""

from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue

try:
    import dask.array as da
except ImportError:
    da = cast(ModuleType, None)

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("dask")
class DaskGenerator(PythonStringGenerator):
    """Generate Dask python code from IR."""

    @classmethod
    def get_memory_info(cls, device: EagerValue = None) -> dict[str, int]:
        """Retrieve current and peak host RSS memory telemetry using stdlib resource.

        Args:
            device (EagerValue): Optional device identifier (unused for host RSS).

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import resource

        rss: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"current": rss, "peak": rss}

    @classmethod
    def initialize_distributed(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Initialize distributed context.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        import dask.distributed as dask_dist

        client_cls = getattr(dask_dist, "Client")
        backend_config = kwargs.get("backend_config")
        if backend_config and hasattr(backend_config, "options") and backend_config.options and "scheduler_address" in backend_config.options:
            client_cls(backend_config.options["scheduler_address"])
        else:
            client_cls()

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
        """Construct and optimize a static Dask computation task graph and return an execution callable.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (EagerValue): Optional compiler arguments ('optimize_graph').

        Returns:
            EagerValue: Execution callable evaluating the optimized Dask task graph.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def forward_fn(*fn_args: EagerValue) -> EagerValue:
            """Evaluate the graph with Dask arrays.

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
            evaluated = evaluate_graph(graph, inputs=cast("dict[str, EvalValue]", inputs))
            if hasattr(graph, "outputs") and graph.outputs:
                if len(graph.outputs) == 1:
                    return cast(EagerValue, evaluated.get(graph.outputs[0]))
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return cast(EagerValue, evaluated)

        try:
            import dask
            import dask.array as da

            delayed_fn = getattr(dask, "delayed")(forward_fn)

            def dask_aot_runner(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
                """Execute optimized Dask task graph.

                Args:
                    *w_args (EagerValue): Input tensors.
                    **w_kwargs (EagerValue): Keyword arguments.

                Returns:
                    EagerValue: Computed outputs.
                """
                da_args = [getattr(da, "from_array")(a.data if isinstance(a, Tensor) else a) for a in w_args]
                task = delayed_fn(*da_args)
                return cast(EagerValue, getattr(dask, "compute")(task)[0])

            return dask_aot_runner
        except Exception:
            return forward_fn
