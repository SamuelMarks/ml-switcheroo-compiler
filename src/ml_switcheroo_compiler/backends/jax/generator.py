# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""JAX/Flax Target Emission."""

import os
from typing import Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.jax.generator_mixins import (
    JaxControlFlowVisitor,
    JaxDistributedVisitor,
    JaxMathVisitor,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("jax")
class JAXCodeGenerator(BaseGenerator):
    """JAX code generator."""

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> object:
        """Load.

        Args:
            filepath (str): The filepath parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
            encoding (str): The encoding parameter.

        Returns:
            object: Result.
        """
        import jax.numpy as jnp

        return jnp.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

    @classmethod
    def save(cls: type, file: str, arr: object, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr: The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import jax.numpy as jnp

        jnp.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        jnp.savez(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args: Positional args.
            **kwds: Keyword args.
        """
        import jax.numpy as jnp

        jnp.savez_compressed(file, *args, **kwds)

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

    def _format_zeros_like(self, op: str, kwargs: dict[str, object]) -> str:
        """Evaluate _format_zeros_like operation.

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

    def _format_full(self, kwargs: dict[str, object]) -> str:
        """Evaluate _format_full operation.

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

    def get_ops_map(self, kwargs: dict[str, object]) -> dict[str, str]:
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
        """Evaluate _emit_constant_assignment operation.

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
        self.indent_level
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs) -> object:")
        self.indent_level = self.indent_level + 1

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into an ahead-of-time (AOT) compiled executable using JAX JIT and XLA lowering.

        Args:
            graph (IRGraph): The target computational graph to compile.
            **kwargs (object): Optional compilation options, such as 'sample_inputs'.

        Returns:
            object: AOT execution callable wrapping the JAX compiled executable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        try:
            import jax
            import jax.numpy as jnp

            scope: dict[str, object] = {}
            exec(self.generate(), scope)
            apply_model_fn = scope.get("apply_model")
            if apply_model_fn is None or not callable(apply_model_fn):
                raise ValueError("Generated JAX module is missing apply_model.")

            def jax_forward(*fn_args: object) -> object:
                """Execute generated JAX model function.

                Args:
                    *fn_args (object): Positional input arguments.

                Returns:
                    object: Computed outputs.
                """
                return apply_model_fn(None, *fn_args)

            sample_inputs = kwargs.get("sample_inputs")
            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)):
                sample_args = [jnp.asarray(x.data if isinstance(x, Tensor) else x) for x in sample_inputs]
                lowered = jax.jit(jax_forward).lower(*sample_args)
                compiled = lowered.compile()

                def aot_executable(*w_args: object, **w_kwargs: object) -> object:
                    """Execute the AOT-compiled JAX executable artifact.

                    Args:
                        *w_args (object): Input tensors.
                        **w_kwargs (object): Keyword arguments.

                    Returns:
                        object: Result from the compiled executable.
                    """
                    jax_args = [jnp.asarray(a.data if isinstance(a, Tensor) else a) for a in w_args]
                    return compiled(*jax_args)

                return aot_executable

            jitted = jax.jit(jax_forward)

            def jit_executable(*w_args: object, **w_kwargs: object) -> object:
                """Execute JIT-compiled graph callable.

                Args:
                    *w_args (object): Input tensors.
                    **w_kwargs (object): Keyword arguments.

                Returns:
                    object: Result from the JIT-compiled function.
                """
                jax_args = [jnp.asarray(a.data if isinstance(a, Tensor) else a) for a in w_args]
                return jitted(*jax_args)

            return jit_executable
        except Exception:
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            def fallback_forward(*fn_args: object) -> object:
                """Execute interpreter graph evaluation fallback.

                Args:
                    *fn_args (object): Positional input arguments.

                Returns:
                    object: Evaluated output tensors.
                """
                input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
                inputs: dict[str, object] = {}
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

            return fallback_forward


JaxGenerator = JAXCodeGenerator
