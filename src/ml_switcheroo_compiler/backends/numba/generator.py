"""Numba code generator for JIT-compiled CPU acceleration."""

from collections.abc import Callable
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("numba")
class NumbaGenerator(PythonStringGenerator):
    """Generate JIT-accelerated Python/Numba code from IR."""

    def __init__(
        self,
        graph: IRGraph,
        fastmath: bool = True,
        parallel: bool = False,
        nogil: bool = False,
        emit_safe_fallback: bool = True,
    ) -> None:
        """Initialize the Numba code generator.

        Args:
            graph (IRGraph): The IR graph to generate code for.
            fastmath (bool): Whether to enable fastmath optimization in Numba JIT.
            parallel (bool): Whether to enable auto-parallelization in Numba JIT.
            nogil (bool): Whether to release GIL during execution.
            emit_safe_fallback (bool): Whether to emit safe fallback wrapper when numba is absent.
        """
        super().__init__(graph)
        self.fastmath: bool = fastmath
        self.parallel: bool = parallel
        self.nogil: bool = nogil
        self.emit_safe_fallback: bool = emit_safe_fallback
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting NumPy/Numba operations.

        Returns:
            str: The prefix 'np'.
        """
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helpers.
        """
        res: list[str] = []
        return res

    def _format_jit_options(self) -> str:
        """Format the keyword arguments for nb.njit.

        Returns:
            str: Comma-separated options string inside parentheses.
        """
        opts: list[str] = []
        if self.fastmath:
            opts.append("fastmath=True")
        if self.parallel:
            opts.append("parallel=True")
        if self.nogil:
            opts.append("nogil=True")
        return f"({', '.join(opts)})" if opts else "()"

    def generate(self) -> str:
        """Generate the complete JIT-compiled Python/Numba script.

        Returns:
            str: The generated Python/Numba script string.
        """
        self.code = [self.header]
        self.add_line("import numpy as np")
        self.add_line("try:")
        self.indent_level += 1
        self.add_line("import numba as nb")
        self.indent_level -= 1
        self.add_line("except ImportError:")
        self.indent_level += 1
        self.add_line("nb = None")
        self.indent_level -= 1
        self.add_line("")

        jit_opts: str = self._format_jit_options()
        if self.emit_safe_fallback:
            self.add_line(f"_njit = nb.njit{jit_opts} if nb is not None else (lambda fn: fn)")
            self.add_line("")
            self.add_line("@_njit")
        else:
            self.add_line(f"@nb.njit{jit_opts}")

        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def compile_fn(self) -> Callable[..., tuple[np.ndarray, ...]]:
        """Compile the generated Python/Numba script into an executable function.

        Returns:
            Callable[..., tuple[np.ndarray, ...]]: Executable compiled function.
        """
        source: str = self.generate()
        namespace: dict[str, object] = {}
        exec(source, namespace)  # noqa: S102
        func: Optional[object] = namespace.get(self._func_name)
        if not callable(func):
            msg = f"Generated code did not produce callable '{self._func_name}'."
            raise RuntimeError(msg)
        return func

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes emitting NumPy operations compiled by Numba.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): The input variable names.
            **kwargs (object): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)
