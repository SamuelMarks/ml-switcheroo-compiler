# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""JAX/Flax Target Emission."""

import os
from typing import Any

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.jax.generator_mixins import (
    JaxAudioVisitor,
    JaxControlFlowVisitor,
    JaxDistributedVisitor,
    JaxMathVisitor,
    JaxVisionVisitor,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("jax")
class JAXCodeGenerator(JaxMathVisitor, JaxControlFlowVisitor, JaxVisionVisitor, JaxAudioVisitor, JaxDistributedVisitor, BaseGenerator):
    """JAX code generator."""

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> Any:
        """Load.

        Args:
            filepath (str): The filepath parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
            encoding (str): The encoding parameter.

        Returns:
            Any: Result.
        """
        import jax.numpy as jnp

        return jnp.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

    @classmethod
    def save(cls: type, file: str, arr: Any, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr (Any): The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import jax.numpy as jnp

        jnp.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: Any, **kwds: Any) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args (Any): Positional args.
            **kwds (Any): Keyword args.
        """
        import jax.numpy as jnp

        jnp.savez(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: Any, **kwds: Any) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args (Any): Positional args.
            **kwds (Any): Keyword args.
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
                JaxAudioVisitor(),
                JaxControlFlowVisitor(generator=self),
                JaxMathVisitor(generator=self),
                JaxVisionVisitor(generator=self),
                JaxDistributedVisitor(generator=self),
            ]
        )

    def _format_zeros_like(self, op: str, kwargs: dict[str, Any]) -> str:
        """Evaluate _format_zeros_like operation.

        Args:
            op (str): The op parameter.
            kwargs (dict[str, Any]): The kwargs parameter.

        Returns:
            str: Result.
        """
        res: str = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: dict[str, Any]) -> str:
        """Evaluate _format_full operation.

        Args:
            kwargs (dict[str, Any]): The kwargs parameter.

        Returns:
            str: Result.
        """
        res: str = "jnp.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def generate(self) -> str:
        """Generate code using strict AST construction (CST) from a base NumPy string."""
        from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
        from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator

        gen: NumpyGenerator = NumpyGenerator(self.graph)
        base_code: str = gen.generate()
        transpiled: Any = transpile_source(base_code, target_framework="jax")
        return str(transpiled)

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
            str: Result.
        """
        return "jnp"

    def get_ops_map(self, kwargs: dict[str, Any]) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs (dict[str, Any]): Operation kwargs.

        Returns:
            dict[str, str]: Dictionary mapping operation type to format string.
        """
        ops: dict[str, str] = super().get_ops_map(kwargs)
        ops["Zeros"] = self._format_zeros_like("zeros", kwargs)
        ops["Ones"] = self._format_zeros_like("ones", kwargs)
        ops["Full"] = self._format_full(kwargs)
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
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs) -> Any:")
        self.indent_level += 1
