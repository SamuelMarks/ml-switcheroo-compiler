"""CuPy code generator and eager execution backend."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import (
    EagerBackendAdapter,
    StaticEagerBackendAdapter,
    adapt_backend_module,
)

try:
    import cupy as cp
except ImportError:
    cp = None

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("cupy")
class CupyGenerator(PythonStringGenerator):
    """Generate CuPy python code from IR."""

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> BackendArg:
        """Allocate a zero-initialized CuPy array.

        Args:
            shape (tuple[int, ...]): Target array shape.

        Returns:
            BackendArg: Zero-initialized CuPy array.
        """
        from .types import zeros

        return zeros(cls, shape)

    @classmethod
    def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
        """Construct a CuPy array from ``data``.

        Args:
            data (BackendArg): Input array or scalar data.
            dtype (BackendArg | None): Optional target data type.

        Returns:
            BackendArg: Constructed CuPy array.
        """
        from .types import array

        return array(cls, data, dtype)

    @classmethod
    def asarray(cls, data: BackendArg) -> BackendArg:
        """Convert ``data`` to a CuPy array view or buffer.

        Args:
            data (BackendArg): Input array or sequence.

        Returns:
            BackendArg: CuPy array.
        """
        from .types import asarray

        return asarray(cls, data)

    @classmethod
    def item(cls, data: BackendArg) -> float:
        """Extract scalar float value from a CuPy array.

        Args:
            data (BackendArg): Scalar CuPy array.

        Returns:
            float: Extracted scalar float.
        """
        from .types import item

        return item(cls, data)

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an IR operation eagerly on CuPy tensors.

        Args:
            op_type (str): Canonical IR operation name.
            *args (BackendArg): Positional tensor operands.
            **kwargs (BackendArg): Keyword operation attributes.

        Returns:
            BackendArg: Evaluated CuPy tensor result.
        """
        from .eager import execute_op

        return execute_op(cls, op_type, *args, **kwargs)

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
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
            list[str]: Computed `list[str]` output.
        """
        res: list[str] = []
        return res

    _import_header = "import cupy as cp"
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
        """Compile IRGraph into an AOT execution callable with CUDA graph capture or compiled kernel.

        Args:
            graph (IRGraph): Target computation graph.
            **kwargs (BackendArg): Optional compiler options ('sample_inputs').

        Returns:
            BackendArg: Callable execution wrapper.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def forward_fn(*fn_args: BackendArg) -> BackendArg:
            """Evaluate the graph with CuPy arrays.

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
            import cupy as cp

            sample_inputs = kwargs.get("sample_inputs")
            if (
                sample_inputs is not None
                and isinstance(sample_inputs, (list, tuple))
                and adapt_backend_module(cp.cuda).resolve_symbol("Graph")[0]
            ):
                cp_samples = [cp.asarray(x) for x in sample_inputs]
                forward_fn(*cp_samples)
                stream = cp.cuda.Stream()
                with stream:
                    graph_record = cp.cuda.Graph()
                    graph_record.begin_capture()
                    out_static = forward_fn(*cp_samples)
                    graph_record.end_capture()
                    instance = graph_record.instantiate()

                def aot_cupy_graph_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                    """Execute captured CuPy CUDA graph.

                    Args:
                        *w_args (BackendArg): Input tensors.
                        **w_kwargs (BackendArg): Keyword arguments.

                    Returns:
                        BackendArg: Computed outputs.
                    """
                    instance.launch(stream)
                    stream.synchronize()
                    return out_static

                return aot_cupy_graph_runner

            def aot_cupy_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                """Execute compiled CuPy graph callable.

                Args:
                    *w_args (BackendArg): Input tensors.
                    **w_kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Computed outputs.
                """
                cp_args = [cp.asarray(a.data if isinstance(a, Tensor) else a) for a in w_args]
                return forward_fn(*cp_args)

            return aot_cupy_runner
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            return forward_fn

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for CuPy.

        Returns:
            EagerBackendAdapter: Concrete CuPy eager backend adapter.
        """
        from .eager import cupy_avgpool2d, cupy_conv2d, cupy_maxpool2d, cupy_resize_bilinear

        return StaticEagerBackendAdapter(
            backend_name="cupy",
            ops={
                "conv2d": cupy_conv2d,
                "maxpool2d": cupy_maxpool2d,
                "avgpool2d": cupy_avgpool2d,
                "resize_bilinear": cupy_resize_bilinear,
                "asarray": cls.asarray,
                "array": cls.array,
                "zeros": cls.zeros,
            },
            module=cp,
        )


CuPyGenerator = CupyGenerator
