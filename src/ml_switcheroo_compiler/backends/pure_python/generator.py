"""Pure Python AST and string code generator."""

from __future__ import annotations

from collections.abc import Callable

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, StaticEagerBackendAdapter
from ml_switcheroo_compiler.backends.pure_python.eager import execute_op
from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor
from ml_switcheroo_compiler.backends.registry import register_backend


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
        **kwargs: BackendArg,
    ) -> str:
        """Generate default pure Python code for an IR node.

        Args:
            node (IRNode): Target IR node.
            input_vars (list[str]): Names of input variables.
            **kwargs (BackendArg): Keyword compiler arguments.

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
        **kwargs: BackendArg,
    ) -> Callable[[PurePythonTensor], PurePythonTensor]:
        """Compile computation graph into an executable pure Python callable.

        Args:
            graph (IRGraph): The IR computation graph.
            **kwargs (BackendArg): Compiler options.

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

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for Pure Python.

        Returns:
            EagerBackendAdapter: Concrete Pure Python eager backend adapter.
        """
        import math

        from ml_switcheroo_compiler.backends.eager_registry import BackendOperand
        from ml_switcheroo_compiler.backends.typeclasses import (
            CoreEagerDictionary,
            CreationDictionary,
            ElementwiseBinaryDictionary,
            ElementwiseUnaryDictionary,
            LinearAlgebraDictionary,
            ReductionDictionary,
            verify_core_eager_typeclass,
        )

        def _pp_fill(val: float, shape: tuple[int, ...], dtype: str | None = None) -> BackendOperand:
            """Allocate a `PurePythonTensor` of `shape` filled with scalar `val`.

            Args:
                val (float): Fill scalar value.
                shape (tuple[int, ...]): Target tensor dimensions.
                dtype (str | None): Optional dtype string.

            Returns:
                BackendOperand: Allocated pure-Python tensor.
            """
            count = 1
            for dim in shape:
                count *= max(dim, 0)
            payload = val if not shape else [val for _ in range(count)]
            res: BackendOperand = PurePythonTensor(payload, shape=shape, dtype=dtype or "float32")
            return res

        def _pp_ingest(data: BackendOperand, dtype: str | None = None) -> PurePythonTensor:
            """Convert host `data` into a `PurePythonTensor`.

            Args:
                data (BackendOperand): Host scalar, list, or `PurePythonTensor`.
                dtype (str | None): Optional dtype string.

            Returns:
                PurePythonTensor: Wrapped `PurePythonTensor`.
            """
            if isinstance(data, PurePythonTensor):
                return data
            if isinstance(data, (int, float)):
                return PurePythonTensor(data, dtype=dtype or "float32")
            if isinstance(data, (list, tuple)):
                flat_nums = [float(str(v)) for v in data]
                return PurePythonTensor(flat_nums, dtype=dtype or "float32")
            return PurePythonTensor(0.0, dtype=dtype or "float32")

        def _pp_bin(op_name: str, lhs: BackendOperand, rhs: BackendOperand) -> BackendOperand:
            """Execute binary PurePython op on `lhs` and `rhs`.

            Args:
                op_name (str): Operation name.
                lhs (BackendOperand): Left operand.
                rhs (BackendOperand): Right operand.

            Returns:
                BackendOperand: Evaluated `PurePythonTensor`.
            """
            a = _pp_ingest(lhs)
            b = _pp_ingest(rhs)
            res: BackendOperand = execute_op(op_name, None, a, b)
            return res

        tc_dict = verify_core_eager_typeclass(
            CoreEagerDictionary[BackendOperand](
                creation=CreationDictionary[BackendOperand](
                    zeros_fn=lambda shape, dtype: _pp_fill(0.0, shape, dtype),
                    ones_fn=lambda shape, dtype: _pp_fill(1.0, shape, dtype),
                    array_fn=_pp_ingest,
                    asarray_fn=_pp_ingest,
                ),
                unary=ElementwiseUnaryDictionary[BackendOperand](
                    abs_fn=lambda x: (
                        x.apply_elementwise(abs) if isinstance(x, PurePythonTensor) else abs(float(str(x)))
                    ),
                    exp_fn=lambda x: x.exp() if isinstance(x, PurePythonTensor) else math.exp(float(str(x))),
                    log_fn=lambda x: x.log() if isinstance(x, PurePythonTensor) else math.log(float(str(x))),
                    sqrt_fn=lambda x: x.sqrt() if isinstance(x, PurePythonTensor) else math.sqrt(float(str(x))),
                    sin_fn=lambda x: x.sin() if isinstance(x, PurePythonTensor) else math.sin(float(str(x))),
                    cos_fn=lambda x: x.cos() if isinstance(x, PurePythonTensor) else math.cos(float(str(x))),
                    tanh_fn=lambda x: x.tanh() if isinstance(x, PurePythonTensor) else math.tanh(float(str(x))),
                ),
                binary=ElementwiseBinaryDictionary[BackendOperand](
                    add_fn=lambda lhs, rhs: _pp_bin("add", lhs, rhs),
                    subtract_fn=lambda lhs, rhs: _pp_bin("subtract", lhs, rhs),
                    multiply_fn=lambda lhs, rhs: _pp_bin("multiply", lhs, rhs),
                    divide_fn=lambda lhs, rhs: _pp_bin("divide", lhs, rhs),
                ),
                linalg=LinearAlgebraDictionary[BackendOperand](
                    matmul_fn=lambda lhs, rhs: _pp_bin("multiply", lhs, rhs),
                    reshape_fn=lambda x, shape: PurePythonTensor(
                        x.to_flat_list() if isinstance(x, PurePythonTensor) else [0.0],
                        shape=shape,
                    ),
                    transpose_fn=lambda x, axes: x,
                ),
                reduction=ReductionDictionary[BackendOperand](
                    sum_fn=lambda x, axis, keepdims: x.sum() if isinstance(x, PurePythonTensor) else float(str(x)),
                    mean_fn=lambda x, axis, keepdims: (
                        sum(float(v) for v in x.to_flat_list()) / max(len(x.to_flat_list()), 1)
                        if isinstance(x, PurePythonTensor)
                        else float(str(x))
                    ),
                ),
            )
        )

        return StaticEagerBackendAdapter(
            backend_name="pure_python",
            ops={
                "abs": abs,
                "sin": math.sin,
                "cos": math.cos,
                "tanh": math.tanh,
                "exp": math.exp,
                "log": math.log,
                "sqrt": math.sqrt,
                "array": cls.array,
                "asarray": cls.asarray,
                "zeros": cls.zeros,
            },
            module=math,
            typeclass_dict=tc_dict,
        )
