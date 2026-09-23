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
        parallel: bool = True,
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

    def visit_WhileLoop(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit Numba JIT-compiled WhileLoop control flow.

        Args:
            node (IRNode): WhileLoop IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        attrs = getattr(node, "attributes", {})
        max_iters = attrs.get("max_iters", 100)
        init_val = input_vars[0] if input_vars else "0"
        res_var = f"v_{node.id.replace('-', '_')}"
        iter_var = f"iter_{node.id.replace('-', '_')}"
        self.add_line(f"{res_var} = {init_val}")
        self.add_line(f"{iter_var} = 0")
        self.add_line(f"while {iter_var} < {max_iters}:")
        self.indent_level += 1
        self.add_line(f"{iter_var} += 1")
        self.add_line(f"{res_var} = {res_var} + 1")
        self.indent_level -= 1
        return res_var

    def visit_ForiLoop(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit Numba JIT-compiled ForiLoop control flow.

        Args:
            node (IRNode): ForiLoop IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        attrs = getattr(node, "attributes", {})
        lower = attrs.get("lower", 0)
        upper = attrs.get("upper", 10)
        step = attrs.get("step", 1)
        init_val = input_vars[0] if input_vars else "0"
        res_var = f"v_{node.id.replace('-', '_')}"
        loop_var = f"idx_{node.id.replace('-', '_')}"
        range_call = f"nb.prange({lower}, {upper}, {step})" if self.parallel else f"range({lower}, {upper}, {step})"
        self.add_line(f"{res_var} = {init_val}")
        self.add_line(f"for {loop_var} in {range_call}:")
        self.indent_level += 1
        self.add_line(f"{res_var} = {res_var} + 1")
        self.indent_level -= 1
        return res_var

    def visit_Sum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel reduction loop with nb.prange or standard np.sum.

        Args:
            node (IRNode): Sum reduction node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        in_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"{out_var} = 0.0")
            self.add_line(f"for _i in nb.prange({in_var}.size):")
            self.indent_level += 1
            self.add_line(f"{out_var} += {in_var}.flat[_i]")
            self.indent_level -= 1
            return out_var
        self.add_line(f"{out_var} = np.sum({in_var})")
        return out_var

    def visit_Prod(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel reduction loop with nb.prange or standard np.prod.

        Args:
            node (IRNode): Prod reduction node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        in_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"{out_var} = 1.0")
            self.add_line(f"for _i in nb.prange({in_var}.size):")
            self.indent_level += 1
            self.add_line(f"{out_var} *= {in_var}.flat[_i]")
            self.indent_level -= 1
            return out_var
        self.add_line(f"{out_var} = np.prod({in_var})")
        return out_var

    def visit_Mean(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel mean reduction or standard np.mean.

        Args:
            node (IRNode): Mean reduction node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        in_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"{out_var}_sum = 0.0")
            self.add_line(f"for _i in nb.prange({in_var}.size):")
            self.indent_level += 1
            self.add_line(f"{out_var}_sum += {in_var}.flat[_i]")
            self.indent_level -= 1
            self.add_line(f"{out_var} = {out_var}_sum / max(1, {in_var}.size)")
            return out_var
        self.add_line(f"{out_var} = np.mean({in_var})")
        return out_var

    def visit_Max(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel max reduction or standard np.max.

        Args:
            node (IRNode): Max reduction node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        in_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"{out_var} = -1e38")
            self.add_line(f"for _i in nb.prange({in_var}.size):")
            self.indent_level += 1
            self.add_line(f"if {in_var}.flat[_i] > {out_var}: {out_var} = {in_var}.flat[_i]")
            self.indent_level -= 1
            return out_var
        self.add_line(f"{out_var} = np.max({in_var})")
        return out_var

    def visit_Min(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel min reduction or standard np.min.

        Args:
            node (IRNode): Min reduction node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        in_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"{out_var} = 1e38")
            self.add_line(f"for _i in nb.prange({in_var}.size):")
            self.indent_level += 1
            self.add_line(f"if {in_var}.flat[_i] < {out_var}: {out_var} = {in_var}.flat[_i]")
            self.indent_level -= 1
            return out_var
        self.add_line(f"{out_var} = np.min({in_var})")
        return out_var

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel matrix multiplication with nb.prange or standard np.matmul.

        Args:
            node (IRNode): MatMul IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        a_var = input_vars[0] if len(input_vars) > 0 else "a"
        b_var = input_vars[1] if len(input_vars) > 1 else "b"
        if self.parallel:
            self.add_line(f"{out_var} = np.zeros(({a_var}.shape[0], {b_var}.shape[1]), dtype={a_var}.dtype)")
            self.add_line(f"for _i in nb.prange({a_var}.shape[0]):")
            self.indent_level += 1
            self.add_line(f"for _j in range({b_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"for _k in range({a_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"{out_var}[_i, _j] += {a_var}[_i, _k] * {b_var}[_k, _j]")
            self.indent_level -= 3
            return out_var
        self.add_line(f"{out_var} = np.matmul({a_var}, {b_var})")
        return out_var

    def visit_BatchMatMul(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel batched matrix multiplication with nb.prange across outer batch.

        Args:
            node (IRNode): BatchMatMul IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        a_var = input_vars[0] if len(input_vars) > 0 else "a"
        b_var = input_vars[1] if len(input_vars) > 1 else "b"
        if self.parallel:
            self.add_line(f"{out_var} = np.zeros(({a_var}.shape[0], {a_var}.shape[1], {b_var}.shape[2]), dtype={a_var}.dtype)")
            self.add_line(f"for _b in nb.prange({a_var}.shape[0]):")
            self.indent_level += 1
            self.add_line(f"for _i in nb.prange({a_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"for _j in range({b_var}.shape[2]):")
            self.indent_level += 1
            self.add_line(f"for _k in range({a_var}.shape[2]):")
            self.indent_level += 1
            self.add_line(f"{out_var}[_b, _i, _j] += {a_var}[_b, _i, _k] * {b_var}[_b, _k, _j]")
            self.indent_level -= 4
            return out_var
        self.add_line(f"{out_var} = np.matmul({a_var}, {b_var})")
        return out_var

    def visit_Conv2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel 2D convolution with nb.prange across batch and spatial height.

        Args:
            node (IRNode): Conv2D IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        x_var = input_vars[0] if len(input_vars) > 0 else "x"
        w_var = input_vars[1] if len(input_vars) > 1 else "w"
        if self.parallel:
            self.add_line(f"_h_out = {x_var}.shape[2] - {w_var}.shape[2] + 1")
            self.add_line(f"_w_out = {x_var}.shape[3] - {w_var}.shape[3] + 1")
            self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {w_var}.shape[0], _h_out, _w_out), dtype={x_var}.dtype)")
            self.add_line(f"for _b in nb.prange({x_var}.shape[0]):")
            self.indent_level += 1
            self.add_line("for _h in nb.prange(_h_out):")
            self.indent_level += 1
            self.add_line("for _w in range(_w_out):")
            self.indent_level += 1
            self.add_line(f"for _oc in range({w_var}.shape[0]):")
            self.indent_level += 1
            self.add_line(f"for _ic in range({x_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"for _kh in range({w_var}.shape[2]):")
            self.indent_level += 1
            self.add_line(f"for _kw in range({w_var}.shape[3]):")
            self.indent_level += 1
            self.add_line(f"{out_var}[_b, _oc, _h, _w] += {x_var}[_b, _ic, _h + _kh, _w + _kw] * {w_var}[_oc, _ic, _kh, _kw]")
            self.indent_level -= 7
            return out_var
        self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {w_var}.shape[0], {x_var}.shape[2] - {w_var}.shape[2] + 1, {x_var}.shape[3] - {w_var}.shape[3] + 1), dtype={x_var}.dtype)")
        return out_var

    def visit_MaxPool2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel 2D max pooling with nb.prange across batch and spatial height.

        Args:
            node (IRNode): MaxPool2D IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        x_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"_h_out = {x_var}.shape[2] // 2")
            self.add_line(f"_w_out = {x_var}.shape[3] // 2")
            self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {x_var}.shape[1], _h_out, _w_out), dtype={x_var}.dtype)")
            self.add_line(f"for _b in nb.prange({x_var}.shape[0]):")
            self.indent_level += 1
            self.add_line("for _h in nb.prange(_h_out):")
            self.indent_level += 1
            self.add_line("for _w in range(_w_out):")
            self.indent_level += 1
            self.add_line(f"for _c in range({x_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"{out_var}[_b, _c, _h, _w] = max({x_var}[_b, _c, _h * 2, _w * 2], {x_var}[_b, _c, _h * 2 + 1, _w * 2], {x_var}[_b, _c, _h * 2, _w * 2 + 1], {x_var}[_b, _c, _h * 2 + 1, _w * 2 + 1])")
            self.indent_level -= 4
            return out_var
        self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {x_var}.shape[1], {x_var}.shape[2] // 2, {x_var}.shape[3] // 2), dtype={x_var}.dtype)")
        return out_var

    def visit_AvgPool2D(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Emit parallel 2D average pooling with nb.prange across batch and spatial height.

        Args:
            node (IRNode): AvgPool2D IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional keyword attributes.

        Returns:
            str: Output variable name.
        """
        out_var = f"v_{node.id.replace('-', '_')}"
        x_var = input_vars[0] if input_vars else "x"
        if self.parallel:
            self.add_line(f"_h_out = {x_var}.shape[2] // 2")
            self.add_line(f"_w_out = {x_var}.shape[3] // 2")
            self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {x_var}.shape[1], _h_out, _w_out), dtype={x_var}.dtype)")
            self.add_line(f"for _b in nb.prange({x_var}.shape[0]):")
            self.indent_level += 1
            self.add_line("for _h in nb.prange(_h_out):")
            self.indent_level += 1
            self.add_line("for _w in range(_w_out):")
            self.indent_level += 1
            self.add_line(f"for _c in range({x_var}.shape[1]):")
            self.indent_level += 1
            self.add_line(f"{out_var}[_b, _c, _h, _w] = ({x_var}[_b, _c, _h * 2, _w * 2] + {x_var}[_b, _c, _h * 2 + 1, _w * 2] + {x_var}[_b, _c, _h * 2, _w * 2 + 1] + {x_var}[_b, _c, _h * 2 + 1, _w * 2 + 1]) / 4.0")
            self.indent_level -= 4
            return out_var
        self.add_line(f"{out_var} = np.zeros(({x_var}.shape[0], {x_var}.shape[1], {x_var}.shape[2] // 2, {x_var}.shape[3] // 2), dtype={x_var}.dtype)")
        return out_var
