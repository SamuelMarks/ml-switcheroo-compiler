"""JAX/Flax Target Emission."""

from __future__ import annotations

import os
import typing  # noqa: F401
from types import ModuleType

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    require_callable,
)
from ml_switcheroo_compiler.backends.jax.generator_mixins import (
    JaxControlFlowVisitor,
    JaxDistributedVisitor,
    JaxMathVisitor,
)
from ml_switcheroo_compiler.backends.jax.jaxpr_ingest import ingest_jaxpr
from ml_switcheroo_compiler.backends.protocols import (
    EagerBackendAdapter,
    StaticEagerBackendAdapter,
    adapt_backend_module,
)
from ml_switcheroo_compiler.backends.registry import register_backend


@typing.runtime_checkable
class _DistributedBackendConfigProtocol(typing.Protocol):
    """Protocol for distributed backend configuration objects."""

    options: dict[str, BackendArg] | None
    world_size: int | None
    rank: int | None


@register_backend("jax")
class JAXCodeGenerator(BaseGenerator):
    """JAX code generator."""

    ingest_graph = staticmethod(ingest_jaxpr)

    @classmethod
    def get_memory_info(cls, device: BackendArg = None) -> dict[str, int]:
        """Retrieve current and peak memory telemetry from JAX device stats or stdlib resource.

        Args:
            device (BackendArg): Optional device identifier.

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import jax

        try:
            dev = jax.local_devices()[0]
            stats = dev.memory_stats()
            return {
                "current": int(coerce_int(stats.get("bytes_in_use", 0))),
                "peak": int(coerce_int(stats.get("peak_bytes_in_use", 0))),
            }
        except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
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
        import jax.distributed

        backend_config: BackendArg | _DistributedBackendConfigProtocol = kwargs.get("backend_config")
        dist_mod = jax.distributed
        if isinstance(backend_config, _DistributedBackendConfigProtocol):
            coord = backend_config.options.get("coordinator_address") if backend_config.options else None
            dist_mod.initialize(
                coordinator_address=coord,
                num_processes=backend_config.world_size,
                process_id=backend_config.rank,
            )
        else:
            dist_mod.initialize()

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
            BackendArg: Computed `BackendArg` output.
        """
        import jax.numpy as jnp

        return require_callable(jnp.load)(
            filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding
        )

    @classmethod
    def save(cls: type, file: str, arr: BackendArg, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): Input `file` argument (file).
            arr: Input `arr` argument (arr).
            allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
            fix_imports (bool): Input `fix_imports` argument (fix imports).
        """
        import jax.numpy as jnp

        require_callable(jnp.save)(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez.

        Args:
            file (str): Input `file` argument (file).
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        require_callable(jnp.savez)(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez compressed.

        Args:
            file (str): Input `file` argument (file).
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        savez_comp_fn = adapt_backend_module(jnp).get_symbol("savez_compressed", None)
        if savez_comp_fn is not None:
            require_callable(savez_comp_fn)(file, *args, **kwds)

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
        """
        super().__init__(graph)
        self.visitors.extend(
            [
                *get_shared_ast_visitors(generator=self),
                JaxControlFlowVisitor(generator=self),
                JaxMathVisitor(generator=self),
                JaxDistributedVisitor(generator=self),
            ]
        )

    def _format_zeros_like(self, op: str, kwargs: dict[str, BackendArg]) -> str:
        """Execute format zeros like for compiler processing.

        Args:
            op (str): Input `op` argument (op).
            kwargs: Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        res: str = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: dict[str, BackendArg]) -> str:
        """Execute format full for compiler processing.

        Args:
            kwargs: Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        res: str = "jnp.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def generate(self) -> str:
        """Generate functional JAX code directly from the IR graph.

        Returns:
            str: Generated JAX source code.
        """
        self.code = [self.header]
        self.code.extend(self._resolve_imports())
        self._generate_function_signature()
        self._generate_body("args")
        return "\n".join(self.code)

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
            str: Computed `str` output.
        """
        return "jnp"

    def get_ops_map(self, kwargs: dict[str, BackendArg] | None = None) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            dict[str, str]: Dictionary mapping operation type to format string.
        """
        kw = kwargs if kwargs is not None else {}
        ops = super().get_ops_map(kw)
        ops["Zeros"] = self._format_zeros_like("zeros", kw)
        ops["Ones"] = self._format_zeros_like("ones", kw)
        ops["Full"] = self._format_full(kw)
        ops["Relu"] = "jax.nn.relu({0})"
        ops["Gelu"] = "jax.nn.gelu({0})"
        ops["Silu"] = "jax.nn.silu({0})"
        ops["Softmax"] = "jax.nn.softmax({0})" if "axis" not in kw else "jax.nn.softmax({0}, axis={axis})"
        ops["LogSoftmax"] = "jax.nn.log_softmax({0})" if "axis" not in kw else "jax.nn.log_softmax({0}, axis={axis})"
        ops["Sigmoid"] = "jax.nn.sigmoid({0})"
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Execute emit constant assignment for compiler processing.

        Args:
            var_name (str): Input `var_name` argument (var name).
            val_repr (str): Input `val_repr` argument (val repr).
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Generate file header with module docstrings.

        Returns:
            list[str]: Computed `list[str]` output.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports.

        Returns:
            list[str]: Computed `list[str]` output.
        """
        tmpl_path: str = os.path.join(os.path.dirname(__file__), "jax_prefix.py.tmpl")
        with open(tmpl_path, encoding="utf-8") as f:
            jax_prefix_template: str = f.read()
        return ["import jax", "import jax.numpy as jnp", "import jax.scipy.special", *jax_prefix_template.split("\n")]

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs) -> object:")
        self.indent_level = self.indent_level + 1

    def _build_jax_interpreter_fallback(self, graph: IRGraph) -> BackendArg:
        """Create an interpreter-based graph evaluation callable for JAX AOT fallback.

        Args:
            graph (IRGraph): Target computational graph.

        Returns:
            BackendArg: Fallback evaluation callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def fallback_forward(*fn_args: BackendArg) -> BackendArg:
            """Execute interpreter graph evaluation fallback.

            Args:
                *fn_args (BackendArg): Positional input arguments.

            Returns:
                BackendArg: Evaluated output tensors.
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

        return fallback_forward

    def _wrap_jax_callable(self, target_fn: BackendArg, jnp_mod: BackendArg) -> BackendArg:
        """Wrap a compiled or JIT-wrapped JAX callable to automatically unwrap Tensor inputs.

        Args:
            target_fn (BackendArg): Underlying JAX compiled or JIT callable.
            jnp_mod (BackendArg): Loaded jax.numpy module.

        Returns:
            BackendArg: Wrapped execution function.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        def executable(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
            """Execute the compiled or JIT-wrapped JAX callable.

            Args:
                *w_args (BackendArg): Input tensors.
                **w_kwargs (BackendArg): Keyword arguments.

            Returns:
                BackendArg: Result from the JAX callable.
            """
            del w_kwargs
            asarray_fn = adapt_backend_module(jnp_mod).require_op("asarray")
            jax_args = [asarray_fn(a.data if isinstance(a, Tensor) else a) for a in w_args]
            return require_callable(target_fn)(*jax_args)

        return executable

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into an ahead-of-time (AOT) compiled executable using JAX JIT and XLA lowering.

        Args:
            graph (IRGraph): The target computational graph to compile.
            **kwargs (BackendArg): Optional compilation options, such as 'sample_inputs'.

        Returns:
            BackendArg: AOT execution callable wrapping the JAX compiled executable.

        Raises:
            ValueError: When raised during `_compile_aot_impl` (Generated JAX module is missing apply_model).
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        try:
            import jax
            import jax.numpy as jnp

            from ml_switcheroo_compiler.backends._module_loader import load_generated_code

            scope = load_generated_code(self.generate())
            apply_model_fn = scope.get("apply_model")
            if apply_model_fn is None or not callable(apply_model_fn):
                raise ValueError("Generated JAX module is missing apply_model.")

            def jax_forward(*fn_args: BackendArg) -> BackendArg:
                """Execute generated JAX model function.

                Args:
                    *fn_args (BackendArg): Positional input arguments.

                Returns:
                    BackendArg: Computed outputs.
                """
                return apply_model_fn(None, *fn_args)

            sample_inputs = kwargs.get("sample_inputs")
            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)):
                sample_args = [jnp.asarray(x.data if isinstance(x, Tensor) else x) for x in sample_inputs]
                compiled = jax.jit(jax_forward).lower(*sample_args).compile()
                return self._wrap_jax_callable(compiled, jnp)

            return self._wrap_jax_callable(jax.jit(jax_forward), jnp)
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
            return self._build_jax_interpreter_fallback(graph)

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute operation eagerly on JAX backend.

        Args:
            op_type (str): Operation type.
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Evaluated JAX result.
        """
        from .eager import execute_op as _execute_op

        return _execute_op(cls, op_type, *args, **kwargs)

    @staticmethod
    def all_reduce(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Delegate all_reduce to distributed_collectives.jax_all_reduce.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Reduced tensor result.
        """
        import importlib

        dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
        return require_callable(adapt_backend_module(dc).require_symbol("jax_all_reduce"))(*args, **kwargs)

    @staticmethod
    def all_gather(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Delegate all_gather to distributed_collectives.jax_all_gather.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Gathered tensor result.
        """
        import importlib

        dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
        return require_callable(adapt_backend_module(dc).require_symbol("jax_all_gather"))(*args, **kwargs)

    @staticmethod
    def broadcast(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Delegate broadcast to distributed_collectives.jax_broadcast.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Broadcasted tensor result.
        """
        import importlib

        dc = importlib.import_module("ml_switcheroo_compiler.backends.jax.distributed_collectives")
        return require_callable(adapt_backend_module(dc).require_symbol("jax_broadcast"))(*args, **kwargs)

    @classmethod
    def get_module(cls) -> ModuleType:
        """Return the underlying ``jax.numpy`` runtime module.

        Returns:
            ModuleType: Imported ``jax.numpy`` module.
        """
        import importlib

        return importlib.import_module("jax.numpy")

    @classmethod
    def execute_collective(
        cls,
        op_name: str,
        tensor: BackendArg,
        **kwargs: BackendArg,
    ) -> BackendArg | None:
        """Execute an eager distributed collective via JAX collective primitives.

        Args:
            op_name (str): Canonical collective operation name.
            tensor (BackendArg): Input tensor buffer.
            **kwargs (BackendArg): Collective parameters (``op``, ``mesh_axis``, ``axis``, ``root``).

        Returns:
            BackendArg | None: Evaluated collective tensor, or ``None`` if unavailable/unsupported.
        """
        try:
            from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
                jax_all_gather,
                jax_all_reduce,
                jax_broadcast,
            )

            raw_axis = kwargs.get("axis", 0)
            axis = int(raw_axis) if isinstance(raw_axis, (int, float)) else 0
            raw_root = kwargs.get("root", 0)
            root = int(raw_root) if isinstance(raw_root, (int, float)) else 0
            mesh_axis = str(kwargs.get("mesh_axis", "data"))
            op_str = str(kwargs.get("op", "SUM"))

            if op_name == "AllReduce":
                return jax_all_reduce(tensor, op=op_str, axis_name=mesh_axis)
            if op_name == "AllGather":
                return jax_all_gather(tensor, axis_name=mesh_axis, axis=axis)
            if op_name == "Broadcast":
                return jax_broadcast(tensor, src=root)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            import logging

            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return None

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for JAX (`jax.numpy`).

        Returns:
            EagerBackendAdapter: Concrete JAX eager backend adapter.
        """
        import jax.numpy as jnp

        return StaticEagerBackendAdapter(
            backend_name="jax",
            ops={
                "abs": jnp.abs,
                "add": jnp.add,
                "subtract": jnp.subtract,
                "multiply": jnp.multiply,
                "divide": jnp.divide,
                "sin": jnp.sin,
                "cos": jnp.cos,
                "tanh": jnp.tanh,
                "exp": jnp.exp,
                "log": jnp.log,
                "sqrt": jnp.sqrt,
                "matmul": jnp.matmul,
                "sum": jnp.sum,
                "mean": jnp.mean,
                "reshape": jnp.reshape,
                "transpose": jnp.transpose,
                "asarray": jnp.asarray,
                "array": jnp.array,
                "zeros": jnp.zeros,
                "ones": jnp.ones,
                "empty": jnp.empty,
            },
            dtypes={
                "float16": jnp.float16,
                "float32": jnp.float32,
                "bfloat16": jnp.bfloat16,
                "int8": jnp.int8,
                "int16": jnp.int16,
                "int32": jnp.int32,
                "uint8": jnp.uint8,
                "bool": jnp.bool_,
                "bool_": jnp.bool_,
                "complex64": jnp.complex64,
            },
            submodules={
                "linalg": jnp.linalg,
                "fft": jnp.fft,
            },
            module=jnp,
        )


JaxGenerator = JAXCodeGenerator
