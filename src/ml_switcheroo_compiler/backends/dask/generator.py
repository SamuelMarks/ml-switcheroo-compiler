"""Dask code generator and eager execution backend."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_operand_list,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import (
    EagerBackendAdapter,
    StaticEagerBackendAdapter,
)
from ml_switcheroo_compiler.backends.registry import register_backend

da: ModuleType | None
try:
    import dask.array as _da

    da = _da
except ImportError:
    da = None


@typing.runtime_checkable
class _OptionsConfigProtocol(typing.Protocol):
    """Protocol for backend configs exposing `.options`."""

    options: BackendArg


@register_backend("dask")
class DaskGenerator(PythonStringGenerator):
    """Generate Dask python code from IR."""

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> BackendArg:
        """Allocate a zero-initialized lazy Dask array.

        Args:
            shape (tuple[int, ...]): Target array shape.

        Returns:
            BackendArg: Zero-initialized Dask array.
        """
        from .types import zeros

        return zeros(cls, shape)

    @classmethod
    def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
        """Construct a lazy Dask array from ``data``.

        Args:
            data (BackendArg): Input array or scalar data.
            dtype (BackendArg | None): Optional target data type.

        Returns:
            BackendArg: Constructed Dask array.
        """
        from .types import array

        return array(cls, data, dtype)

    @classmethod
    def asarray(cls, data: BackendArg) -> BackendArg:
        """Convert ``data`` to a lazy Dask array.

        Args:
            data (BackendArg): Input array or sequence.

        Returns:
            BackendArg: Dask array.
        """
        from .types import asarray

        return asarray(cls, data)

    @classmethod
    def item(cls, data: BackendArg) -> float:
        """Extract scalar float value from a Dask array.

        Args:
            data (BackendArg): Scalar Dask array.

        Returns:
            float: Extracted scalar float.
        """
        from .types import item

        return item(cls, data)

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an IR operation lazily on Dask arrays.

        Args:
            op_type (str): Canonical IR operation name.
            *args (BackendArg): Positional tensor operands.
            **kwargs (BackendArg): Keyword operation attributes.

        Returns:
            BackendArg: Lazy Dask array result.
        """
        from .eager import execute_op

        return execute_op(cls, op_type, *args, **kwargs)

    @classmethod
    def get_memory_info(cls, device: BackendArg = None) -> dict[str, int]:
        """Retrieve current and peak host RSS memory telemetry using stdlib resource.

        Args:
            device (BackendArg): Optional device identifier (unused for host RSS).

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import resource

        rss: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"current": rss, "peak": rss}

    @classmethod
    def initialize_distributed(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Initialize distributed context.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).
        """
        import dask.distributed as dask_dist

        client_cls = dask_dist.Client
        backend_config: BackendArg | _OptionsConfigProtocol = kwargs.get("backend_config")
        opts = backend_config.options if isinstance(backend_config, _OptionsConfigProtocol) else None
        if isinstance(opts, dict) and opts and "scheduler_address" in opts:
            client_cls(opts["scheduler_address"])
        else:
            client_cls()

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting Dask array operations.

        Returns:
        str: Computed `str` output.
        """
        return "da"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions.

        Returns:
            list[str]: Computed `list[str]` output.
        """
        res: list[str] = []
        return res

    _import_header = "import dask.array as da"
    _func_name = "evaluate"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Fallback for generic nodes.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Extra attributes.

        Returns:
            str: Generated code.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Construct and optimize a static Dask computation task graph and return an execution callable.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (BackendArg): Optional compiler arguments ('optimize_graph').

        Returns:
            BackendArg: Execution callable evaluating the optimized Dask task graph.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def forward_fn(*fn_args: BackendArg) -> BackendArg:
            """Evaluate the graph with Dask arrays.

            Args:
                *fn_args (BackendArg): Input tensor values.

            Returns:
                BackendArg: Output tensor or collection of tensors.
            """
            input_nodes = [n for n in graph.nodes.values() if n.op_type == "Input"]
            inputs: dict[str, BackendArg] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(fn_args):
                    arg_val = fn_args[i]
                    inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
            evaluated = evaluate_graph(graph, inputs=inputs)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        try:
            import dask
            import dask.array as da

            delayed_fn = require_callable(dask.delayed)(forward_fn)

            def dask_aot_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                """Execute optimized Dask task graph.

                Args:
                    *w_args (BackendArg): Input tensors.
                    **w_kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Computed outputs.
                """
                da_args = [require_callable(da.from_array)(a.data if isinstance(a, Tensor) else a) for a in w_args]
                task = require_callable(delayed_fn)(*da_args)
                return coerce_operand_list(require_callable(dask.compute)(task))[0]

            return dask_aot_runner
        except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return forward_fn

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for Dask (`dask.array`).

        Returns:
            EagerBackendAdapter: Concrete Dask eager backend adapter.
        """
        import dask.array as dask_arr

        return StaticEagerBackendAdapter(
            backend_name="dask",
            ops={
                "abs": dask_arr.abs,
                "add": dask_arr.add,
                "subtract": dask_arr.subtract,
                "multiply": dask_arr.multiply,
                "divide": dask_arr.divide,
                "sin": dask_arr.sin,
                "cos": dask_arr.cos,
                "tanh": dask_arr.tanh,
                "exp": dask_arr.exp,
                "log": dask_arr.log,
                "sqrt": dask_arr.sqrt,
                "matmul": dask_arr.matmul,
                "sum": dask_arr.sum,
                "mean": dask_arr.mean,
                "reshape": dask_arr.reshape,
                "transpose": dask_arr.transpose,
                "asarray": dask_arr.asarray,
                "array": dask_arr.array,
                "zeros": dask_arr.zeros,
                "ones": dask_arr.ones,
                "empty": dask_arr.empty,
            },
            submodules={
                "linalg": dask_arr.linalg,
                "fft": dask_arr.fft,
                "random": dask_arr.random,
            },
            module=dask_arr,
        )
