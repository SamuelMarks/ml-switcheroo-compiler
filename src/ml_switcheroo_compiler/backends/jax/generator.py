"""JAX/Flax Target Emission."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional, cast  # noqa: F401

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.jax.generator_mixins import (
    JaxControlFlowVisitor,
    JaxDistributedVisitor,
    JaxMathVisitor,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("jax")
class JAXCodeGenerator(BaseGenerator):
    """JAX code generator."""

    @classmethod
    def get_memory_info(cls, device: EagerValue = None) -> dict[str, int]:
        """Retrieve current and peak memory telemetry from JAX device stats or stdlib resource.

        Args:
            device (EagerValue): Optional device identifier.

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import jax

        try:
            dev = jax.local_devices()[0]
            stats = dev.memory_stats()
            return {
                "current": int(getattr(stats, "get")("bytes_in_use", 0)),
                "peak": int(getattr(stats, "get")("peak_bytes_in_use", 0)),
            }
        except Exception:
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
        import jax.distributed

        backend_config = kwargs.get("backend_config")
        if backend_config:
            getattr(jax.distributed, "initialize")(coordinator_address=getattr(backend_config, "options").get("coordinator_address") if getattr(backend_config, "options") else None, num_processes=getattr(backend_config, "world_size"), process_id=getattr(backend_config, "rank"))
        else:
            getattr(jax.distributed, "initialize")()

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> EagerValue:
        """Load.

        Args:
            filepath (str): The filepath parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
            encoding (str): The encoding parameter.

        Returns:
            EagerValue: Result.
        """
        import jax.numpy as jnp

        return jnp.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

    @classmethod
    def save(cls: type, file: str, arr: EagerValue, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr: The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import jax.numpy as jnp

        getattr(jnp, "save")(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        getattr(jnp, "savez")(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        getattr(jnp, "savez_compressed")(file, *args, **kwds)

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): The graph parameter.
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

    def _format_zeros_like(self, op: str, kwargs: dict[str, EagerValue]) -> str:
        """Execute format zeros like for compiler processing.

        Args:
            op (str): The op parameter.
            kwargs: The kwargs parameter.

        Returns:
            str: Result.
        """
        res: str = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: dict[str, EagerValue]) -> str:
        """Execute format full for compiler processing.

        Args:
            kwargs: The kwargs parameter.

        Returns:
            str: Result.
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
            str: Result.
        """
        return "jnp"

    def get_ops_map(self, kwargs: dict[str, EagerValue]) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            dict[str, str]: Dictionary mapping operation type to format string.
        """
        ops = super().get_ops_map(kwargs)
        ops["Zeros"] = self._format_zeros_like("zeros", kwargs)
        ops["Ones"] = self._format_zeros_like("ones", kwargs)
        ops["Full"] = self._format_full(kwargs)
        ops["Relu"] = "jax.nn.relu({0})"
        ops["Gelu"] = "jax.nn.gelu({0})"
        ops["Silu"] = "jax.nn.silu({0})"
        ops["Softmax"] = "jax.nn.softmax({0})" if "axis" not in kwargs else "jax.nn.softmax({0}, axis={axis})"
        ops["LogSoftmax"] = "jax.nn.log_softmax({0})" if "axis" not in kwargs else "jax.nn.log_softmax({0}, axis={axis})"
        ops["Sigmoid"] = "jax.nn.sigmoid({0})"
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Execute emit constant assignment for compiler processing.

        Args:
            var_name (str): The var_name parameter.
            val_repr (str): The val_repr parameter.
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Generate file header with module docstrings.

        Returns:
            list[str]: Result.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports.

        Returns:
            list[str]: Result.
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

    def _build_jax_interpreter_fallback(self, graph: IRGraph) -> EagerValue:
        """Create an interpreter-based graph evaluation callable for JAX AOT fallback.

        Args:
            graph (IRGraph): Target computational graph.

        Returns:
            EagerValue: Fallback evaluation callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def fallback_forward(*fn_args: EagerValue) -> EagerValue:
            """Execute interpreter graph evaluation fallback.

            Args:
                *fn_args (EagerValue): Positional input arguments.

            Returns:
                EagerValue: Evaluated output tensors.
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

        return fallback_forward

    def _wrap_jax_callable(self, target_fn: EagerValue, jnp_mod: EagerValue) -> EagerValue:
        """Wrap a compiled or JIT-wrapped JAX callable to automatically unwrap Tensor inputs.

        Args:
            target_fn (EagerValue): Underlying JAX compiled or JIT callable.
            jnp_mod (EagerValue): Loaded jax.numpy module.

        Returns:
            EagerValue: Wrapped execution function.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        def executable(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
            """Execute the compiled or JIT-wrapped JAX callable.

            Args:
                *w_args (EagerValue): Input tensors.
                **w_kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerValue: Result from the JAX callable.
            """
            del w_kwargs
            jax_args = [getattr(jnp_mod, "asarray")(a.data if isinstance(a, Tensor) else a) for a in w_args]
            return cast(Callable[..., EagerValue], target_fn)(*jax_args)

        return executable

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an ahead-of-time (AOT) compiled executable using JAX JIT and XLA lowering.

        Args:
            graph (IRGraph): The target computational graph to compile.
            **kwargs (EagerValue): Optional compilation options, such as 'sample_inputs'.

        Returns:
            EagerValue: AOT execution callable wrapping the JAX compiled executable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        try:
            import jax
            import jax.numpy as jnp

            scope: dict[str, EagerValue] = {}
            exec(self.generate(), scope)
            apply_model_fn = scope.get("apply_model")
            if apply_model_fn is None or not callable(apply_model_fn):
                raise ValueError("Generated JAX module is missing apply_model.")

            def jax_forward(*fn_args: EagerValue) -> EagerValue:
                """Execute generated JAX model function.

                Args:
                    *fn_args (EagerValue): Positional input arguments.

                Returns:
                    EagerValue: Computed outputs.
                """
                return apply_model_fn(None, *fn_args)

            sample_inputs = kwargs.get("sample_inputs")
            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)):
                sample_args = [jnp.asarray(x.data if isinstance(x, Tensor) else x) for x in sample_inputs]
                compiled = jax.jit(jax_forward).lower(*sample_args).compile()
                return self._wrap_jax_callable(compiled, jnp)

            return self._wrap_jax_callable(jax.jit(jax_forward), jnp)
        except Exception:
            return self._build_jax_interpreter_fallback(graph)


JaxGenerator = JAXCodeGenerator
