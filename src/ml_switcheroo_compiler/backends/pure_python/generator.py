"""Pure Python AST and string code generator."""

from __future__ import annotations

from typing import Callable

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.pure_python.eager import execute_op
from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("pure_python")
class PurePythonGenerator(PythonStringGenerator):
    """Generate pure Python standard library code without external framework dependencies."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize PurePythonGenerator.

        Args:
            graph (IRGraph): Target computation graph.
        """
        super().__init__(graph)

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting pure Python math operations.

        Returns:
            str: Module prefix string 'math'.
        """
        return "math"

    def get_helper_functions(self) -> list[str]:
        """Get pure Python helper functions for code generation.

        Returns:
            list[str]: Helper function strings list.
        """
        return []

    _import_header: str = "import math"
    _func_name: str = "evaluate"

    def generic_visit(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: str | int | float | bool,
    ) -> str:
        """Generate default pure Python code for an IR node.

        Args:
            node (IRNode): Target IR node.
            input_vars (list[str]): Names of input variables.
            **kwargs (Union[str, int, float, bool]): Keyword compiler arguments.

        Returns:
            str: Emitted Python expression string.
        """
        del kwargs
        op: str = node.op_type.lower()
        if len(input_vars) == 1:
            if op in ("exp", "log", "sqrt", "sin", "cos", "tanh"):
                return f"{node.id} = math.{op}({input_vars[0]})"
            if op in ("neg", "negative"):
                return f"{node.id} = -{input_vars[0]}"
            return f"{node.id} = {input_vars[0]}"
        if len(input_vars) == 2:
            op_map: dict[str, str] = {
                "add": "+",
                "plus": "+",
                "sub": "-",
                "subtract": "-",
                "mul": "*",
                "multiply": "*",
                "div": "/",
                "divide": "/",
                "truediv": "/",
            }
            sym: str = op_map.get(op, "+")
            return f"{node.id} = {input_vars[0]} {sym} {input_vars[1]}"
        return f"{node.id} = None"

    def _compile_aot_impl(
        self,
        graph: IRGraph,
        **kwargs: object,
    ) -> Callable[[PurePythonTensor], PurePythonTensor]:
        """Compile computation graph into an executable pure Python callable.

        Args:
            graph (IRGraph): The IR computation graph.
            **kwargs (Union[str, int, float, bool]): Compiler options.

        Returns:
            Callable[[PurePythonTensor], PurePythonTensor]: Executable pure Python wrapper.
        """
        del kwargs

        def _runner(input_tensor: PurePythonTensor) -> PurePythonTensor:
            """Execute compiled computation graph against an input tensor.

            Args:
                input_tensor (PurePythonTensor): Input tensor.

            Returns:
                PurePythonTensor: Final computed output tensor.
            """
            env: dict[str, PurePythonTensor] = {}
            last_res: PurePythonTensor = input_tensor
            for node in graph.nodes.values():
                if node.op_type == "Input":
                    env[node.id] = input_tensor
                elif node.op_type == "Output":
                    if node.inputs and node.inputs[0] in env:
                        return env[node.inputs[0]]
                    return last_res
                elif node.op_type in ("Parameter", "Constant"):
                    env[node.id] = PurePythonTensor(0.0)
                else:
                    node_args: list[PurePythonTensor] = [env[inp] for inp in node.inputs if inp in env]
                    if not node_args:
                        node_args = [input_tensor]
                    res: PurePythonTensor | float | int = execute_op(node.op_type, None, *node_args)
                    if isinstance(res, PurePythonTensor):
                        env[node.id] = res
                    else:
                        env[node.id] = PurePythonTensor(res)
                    last_res = env[node.id]
            return last_res

        return _runner
