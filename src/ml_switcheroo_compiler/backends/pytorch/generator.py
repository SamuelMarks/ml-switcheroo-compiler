"""Operations and utilities for `ml_switcheroo_compiler.backends.pytorch.generator`."""

from __future__ import annotations

import logging
import os
from types import ModuleType

from ml_switcheroo_ir import IRGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import (
    EagerBackendAdapter,
    StaticEagerBackendAdapter,
    adapt_backend_module,
)
from ml_switcheroo_compiler.backends.registry import register_backend

from .distributed_collectives import (
    pytorch_all_gather,
    pytorch_all_reduce,
    pytorch_broadcast,
    pytorch_reduce_scatter,
)
from .eager import execute_op as _pytorch_execute_op
from .fx_ingest import ingest_torch_fx
from .pytorch_mixins import (  # noqa: F401
    PyTorchDistributedVisitor,
    PyTorchLinalgMixin,
    PyTorchNNMixin,
    PyTorchScatterVisitor,
)
from .types import array as _pytorch_array
from .types import asarray as _pytorch_asarray
from .types import item as _pytorch_item
from .types import zeros as _pytorch_zeros


def _pt_all_reduce(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Delegate ``all_reduce`` to ``distributed_collectives.pytorch_all_reduce``.

    Args:
        *args (BackendArg): Positional collective arguments.
        **kwargs (BackendArg): Keyword collective arguments.

    Returns:
        BackendArg: All-reduced tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return require_callable(adapt_backend_module(dc).require_symbol("pytorch_all_reduce"))(*args, **kwargs)


def _pt_all_gather(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Delegate ``all_gather`` to ``distributed_collectives.pytorch_all_gather``.

    Args:
        *args (BackendArg): Positional collective arguments.
        **kwargs (BackendArg): Keyword collective arguments.

    Returns:
        BackendArg: All-gathered tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return require_callable(adapt_backend_module(dc).require_symbol("pytorch_all_gather"))(*args, **kwargs)


def _pt_reduce_scatter(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Delegate ``reduce_scatter`` to ``distributed_collectives.pytorch_reduce_scatter``.

    Args:
        *args (BackendArg): Positional collective arguments.
        **kwargs (BackendArg): Keyword collective arguments.

    Returns:
        BackendArg: Reduce-scattered tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return require_callable(adapt_backend_module(dc).require_symbol("pytorch_reduce_scatter"))(*args, **kwargs)


def _pt_broadcast(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Delegate ``broadcast`` to ``distributed_collectives.pytorch_broadcast``.

    Args:
        *args (BackendArg): Positional collective arguments.
        **kwargs (BackendArg): Keyword collective arguments.

    Returns:
        BackendArg: Broadcasted tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return require_callable(adapt_backend_module(dc).require_symbol("pytorch_broadcast"))(*args, **kwargs)


_ = (pytorch_all_gather, pytorch_all_reduce, pytorch_broadcast, pytorch_reduce_scatter)


@register_backend("pytorch")
class PyTorchCodeGenerator(PyTorchLinalgMixin, PyTorchNNMixin, ClassBasedGenerator):
    """PyTorch code generator."""

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> BackendArg:
        """Create a zero-initialized PyTorch tensor.

        Args:
            shape (tuple[int, ...]): Target shape tuple.

        Returns:
            BackendArg: Zero-initialized tensor.
        """
        return _pytorch_zeros(cls, shape)

    @classmethod
    def array(cls, data: BackendArg, dtype: BackendArg = None) -> BackendArg:
        """Construct a PyTorch tensor copy from ``data``.

        Args:
            data (BackendArg): Source array or sequence.
            dtype (BackendArg): Optional data type.

        Returns:
            BackendArg: Constructed PyTorch tensor.
        """
        return _pytorch_array(cls, data, dtype=dtype)

    @classmethod
    def asarray(cls, data: BackendArg) -> BackendArg:
        """Convert ``data`` into a PyTorch tensor without unnecessary copying.

        Args:
            data (BackendArg): Source array or sequence.

        Returns:
            BackendArg: Converted PyTorch tensor.
        """
        return _pytorch_asarray(cls, data)

    @classmethod
    def item(cls, data: BackendArg) -> float:
        """Extract a Python float scalar from a 0-D or 1-element PyTorch tensor.

        Args:
            data (BackendArg): Scalar PyTorch tensor.

        Returns:
            float: Extracted Python float scalar.
        """
        return _pytorch_item(cls, data)

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an IR operation eagerly on the PyTorch backend.

        Args:
            op_type (str): Canonical operation name.
            *args (BackendArg): Positional operands.
            **kwargs (BackendArg): Keyword operation attributes.

        Returns:
            BackendArg: Evaluated PyTorch tensor or scalar.
        """
        return _pytorch_execute_op(cls, op_type, *args, **kwargs)

    all_reduce = staticmethod(_pt_all_reduce)
    all_gather = staticmethod(_pt_all_gather)
    reduce_scatter = staticmethod(_pt_reduce_scatter)
    broadcast = staticmethod(_pt_broadcast)
    ingest_graph = staticmethod(ingest_torch_fx)

    @classmethod
    def get_memory_info(cls, device: BackendArg = None) -> dict[str, int]:
        """Retrieve current and peak memory telemetry from torch.cuda or stdlib resource.

        Args:
            device (BackendArg): Optional device identifier.

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import resource

        import torch

        if torch.cuda.is_available() and (device is None or "cuda" in str(device)):
            dev = device if device is not None else "cuda"
            cuda_mod = torch.cuda
            return {
                "current": int(coerce_int(cuda_mod.memory_allocated(dev))),
                "peak": int(coerce_int(cuda_mod.max_memory_allocated(dev))),
            }
        rss: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"current": rss, "peak": rss}

    @classmethod
    def initialize_distributed(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Initialize distributed context.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).
        """
        import torch.distributed

        backend_config = kwargs.get("backend_config")
        if not torch.distributed.is_initialized() and backend_config:
            torch.distributed.init_process_group(
                backend=adapt_backend_module(backend_config).require_symbol("backend"),
                init_method=adapt_backend_module(backend_config).require_symbol("init_method"),
                world_size=adapt_backend_module(backend_config).require_symbol("world_size"),
                rank=adapt_backend_module(backend_config).require_symbol("rank"),
            )

    _base_class_name: str = "nn.Module"

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the PyTorch code generator with the given computation graph.

        Args:
            graph (IRGraph): The computation graph to compile.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self), PyTorchScatterVisitor()])

    def visit_PowerIteration(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate PyTorch code for a power iteration operation.

        Args:
            node (LogicalNode): The IR node representing the power iteration.
            input_vars (list[str]): The names of the input variables.
            **kwargs (BackendArg): Additional arguments such as iteration counts.

        Returns:
            str: The generated PyTorch code for power iteration.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"pt_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate PyTorch code for a ragged dot product operation.

        Args:
            node (LogicalNode): The IR node representing the ragged dot.
            input_vars (list[str]): The names of the input matrices.
            **kwargs (BackendArg): Additional operation attributes.

        Returns:
            str: The generated PyTorch code for the ragged dot operation.
        """
        return f"pt_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate PyTorch code for an Einstein summation operation.

        Args:
            node (LogicalNode): The IR node representing the einsum operation.
            input_vars (list[str]): The input variable names.
            **kwargs (BackendArg): Additional parameters, including the equation string.

        Returns:
            str: The generated PyTorch code using torch.einsum.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"torch.einsum('{eq}', {args_str})"

    def generate(self) -> str:
        """Generate PyTorch code directly from the IR graph as an nn.Module class.

        Returns:
            str: The generated PyTorch nn.Module source code.
        """
        return super().generate()

    def get_fallback_prefix(self) -> str:
        """Retrieve the fallback library prefix used for missing generic operations.

        Returns:
            str: The fallback prefix string ("torch").
        """
        return "torch"

    def get_fallback_axis_kwarg(self) -> str:
        """Retrieve the keyword argument name used for specifying dimensions.

        Returns:
            str: The axis keyword argument name ("dim").
        """
        return "dim"

    def get_fallback_keepdims_kwarg(self) -> str:
        """Retrieve the keyword argument name used for keeping dimensions.

        Returns:
            str: The keepdims keyword argument name ("keepdim").
        """
        return "keepdim"

    def _get_math_ops(self, kwargs: BackendArg) -> dict[str, str]:
        """Get PyTorch mathematical operation templates.

        Args:
            kwargs (BackendArg): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        del kwargs
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        return {k: v.ast_template for k, v in load_backend_mappings("pytorch").operations.items() if v.ast_template}

    def _get_creation_ops(self, kwargs: BackendArg) -> dict[str, str]:
        """Get PyTorch tensor creation operation templates.

        Args:
            kwargs (BackendArg): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        return self._get_math_ops(kwargs)

    def _get_array_ops(self, kwargs: BackendArg) -> dict[str, str]:
        """Get PyTorch array manipulation operation templates.

        Args:
            kwargs (BackendArg): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        return self._get_math_ops(kwargs)

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Emit the code to assign a constant parameter to a local variable.

        Args:
            var_name (str): Input `var_name` argument (var name).
            val_repr (str): Input `val_repr` argument (val repr).
        """
        self.add_line(f"{var_name} = self.{var_name}")

    def _get_prefix_code(self) -> list[str]:
        """Generate the prefix code required for the PyTorch module, including imports and utilities.

        Returns:
            list[str]: A list of string code lines to be emitted before the main class.
        """
        tmpl_path = os.path.join(os.path.dirname(__file__), "pytorch_prefix.py.tmpl")
        with open(tmpl_path) as f:
            pt_prefix_template = f.read()
        return ["import torch", "import torch.nn as nn", *pt_prefix_template.splitlines()]

    def _emit_init_body(self) -> bool:
        """Emit the initialization code for the PyTorch module's parameters.

        Returns:
            bool: True if any parameters were registered in the module, False otherwise.
        """
        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(f"self.register_parameter('{var_name}', nn.Parameter(torch.tensor({val_repr})))")
                has_params = True
        return has_params

    @classmethod
    def load(
        cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII"
    ) -> BackendArg:
        """Load.

        Args:
        filepath (str): Input `filepath` argument (filepath).
        allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
        fix_imports (bool): Input `fix_imports` argument (fix imports).
        encoding (str): Input `encoding` argument (encoding).

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        import torch

        return require_callable(torch.load)(filepath, weights_only=not allow_pickle)

    @classmethod
    def save(cls: type, file: str, arr: BackendArg, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): Input `file` argument (file).
            arr (BackendArg): Input `arr` argument (arr).
            allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
            fix_imports (bool): Input `fix_imports` argument (fix imports).
        """
        import torch

        torch.save(arr, file)

    @classmethod
    def savez(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez.

        Args:
            file (str): Input `file` argument (file).
            *args (BackendArg): Positional args.
            **kwds (BackendArg): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez compressed.

        Args:
            file (str): Input `file` argument (file).
            *args (BackendArg): Positional args.
            **kwds (BackendArg): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)

    def _create_pytorch_graph_evaluator(self, graph: IRGraph) -> BackendArg:
        """Create an interpreter-backed callable evaluating the IR graph for PyTorch.

        Args:
            graph (IRGraph): Target computational graph.

        Returns:
            BackendArg: Graph evaluation callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        class GraphModule:
            """Callable module wrapper evaluating the IR graph."""

            def __call__(self, *fn_args: BackendArg, **fn_kwargs: BackendArg) -> BackendArg:
                """Forward execution call.

                Args:
                    *fn_args (BackendArg): Input tensors.
                    **fn_kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Computed outputs.
                """
                del fn_kwargs
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

        return GraphModule()

    def _wrap_pytorch_runner(self, compiled_fn: BackendArg, torch_mod: BackendArg) -> BackendArg:
        """Wrap a compiled or traced PyTorch callable to convert inputs via torch.as_tensor.

        Args:
            compiled_fn (BackendArg): Compiled or traced PyTorch function.
            torch_mod (BackendArg): Loaded torch module.

        Returns:
            BackendArg: Wrapped execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        def runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
            """Execute compiled PyTorch callable.

            Args:
                *w_args (BackendArg): Input tensors.
                **w_kwargs (BackendArg): Keyword arguments.

            Returns:
                BackendArg: Output tensor data.
            """
            del w_kwargs
            as_tensor_fn = adapt_backend_module(torch_mod).require_op("as_tensor")
            pt_args = [as_tensor_fn(a.data if isinstance(a, Tensor) else a) for a in w_args]
            return require_callable(compiled_fn)(*pt_args)

        return runner

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into an optimized executable callable using TorchScript or torch.compile.

        Args:
            graph (IRGraph): Target computational graph to compile.
            **kwargs (BackendArg): Optional compiler options ('mode', 'backend', 'sample_inputs').

        Returns:
            BackendArg: Optimized executable callable wrapping the compiled PyTorch module.
        """
        mod = self._create_pytorch_graph_evaluator(graph)

        try:
            import torch

            backend = str(kwargs.get("backend", "inductor"))
            mode = str(kwargs.get("mode", "default"))
            sample_inputs = kwargs.get("sample_inputs")

            if adapt_backend_module(torch).has_symbol("compile"):
                try:
                    compiled_fn = torch.compile(mod, backend=backend, mode=mode)
                    return self._wrap_pytorch_runner(compiled_fn, torch)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)):
                pt_samples = [torch.as_tensor(x) for x in sample_inputs]
                traced = torch.jit.trace(mod, pt_samples)
                return self._wrap_pytorch_runner(traced, torch)

            return mod
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
            return mod

    @classmethod
    def get_module(cls) -> ModuleType:
        """Return the underlying PyTorch runtime module.

        Returns:
            ModuleType: Imported ``torch`` module.
        """
        import importlib

        return importlib.import_module("torch")

    @classmethod
    def execute_collective(
        cls,
        op_name: str,
        tensor: BackendArg,
        **kwargs: BackendArg,
    ) -> BackendArg | None:
        """Execute an eager distributed collective via PyTorch distributed primitives.

        Args:
            op_name (str): Canonical collective operation name.
            tensor (BackendArg): Input tensor buffer.
            **kwargs (BackendArg): Collective parameters (``op``, ``axis``, ``scatter_dim``, ``root``).

        Returns:
            BackendArg | None: Evaluated collective tensor, or ``None`` if unavailable/unsupported.
        """
        try:
            from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
                pytorch_all_gather,
                pytorch_all_reduce,
                pytorch_broadcast,
                pytorch_reduce_scatter,
            )

            raw_axis = kwargs.get("axis", 0)
            axis = int(raw_axis) if isinstance(raw_axis, (int, float)) else 0
            raw_scatter = kwargs.get("scatter_dim", 0)
            scatter_dim = int(raw_scatter) if isinstance(raw_scatter, (int, float)) else 0
            raw_root = kwargs.get("root", 0)
            root = int(raw_root) if isinstance(raw_root, (int, float)) else 0
            op_str = str(kwargs.get("op", "SUM"))

            if op_name == "AllReduce":
                return pytorch_all_reduce(tensor, op=op_str)
            if op_name == "AllGather":
                return pytorch_all_gather(tensor, axis=axis)
            if op_name == "ReduceScatter":
                return pytorch_reduce_scatter(tensor, op=op_str, scatter_dim=scatter_dim)
            if op_name == "Broadcast":
                return pytorch_broadcast(tensor, src=root)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return None

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for PyTorch.

        Returns:
            EagerBackendAdapter: Concrete PyTorch eager backend adapter.
        """
        import torch

        return StaticEagerBackendAdapter(
            backend_name="pytorch",
            ops={
                "abs": torch.abs,
                "add": torch.add,
                "sub": torch.sub,
                "subtract": torch.subtract,
                "mul": torch.mul,
                "multiply": torch.multiply,
                "div": torch.div,
                "divide": torch.divide,
                "sin": torch.sin,
                "cos": torch.cos,
                "tanh": torch.tanh,
                "exp": torch.exp,
                "log": torch.log,
                "sqrt": torch.sqrt,
                "matmul": torch.matmul,
                "sum": torch.sum,
                "mean": torch.mean,
                "reshape": torch.reshape,
                "transpose": torch.permute,
                "asarray": torch.asarray,
                "array": torch.tensor,
                "zeros": torch.zeros,
                "ones": torch.ones,
                "empty": torch.empty,
            },
            dtypes={
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "uint8": torch.uint8,
                "bool": torch.bool,
                "complex64": torch.complex64,
                "complex128": torch.complex128,
            },
            submodules={
                "linalg": torch.linalg,
                "fft": torch.fft,
                "nn": torch.nn,
            },
            module=torch,
        )


PyTorchGenerator = PyTorchCodeGenerator
