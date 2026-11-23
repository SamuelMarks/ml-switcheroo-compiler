"""Operations and utilities for `ml_switcheroo_compiler.backends.numpy.generator`."""

from __future__ import annotations

from types import ModuleType

from ml_switcheroo_ir import IRGraph, IRNode, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_shape
from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike

from .numpy_mixins import NumpyScatterVisitor


class NumpyTypeTranslator:
    """Utility for Numpy type mappings."""

    @staticmethod
    def get_fallback_prefix() -> str:
        """Retrieve the default fallback prefix string for NumPy operations.

        Returns:
            str: The fallback prefix string 'np'.
        """
        return "np"


class NumpyASTVisitor:
    """Visitor methods for Numpy AST traversal."""

    @classmethod
    def _format_kwargs(cls, kwargs: dict[str, BackendArg]) -> str:
        """Format keyword arguments for NumPy code emission.

        Args:
            kwargs (dict[str, BackendArg]): Keyword arguments dictionary.

        Returns:
            str: Comma-separated key=value formatting string.
        """
        filtered_kwargs: dict[str, BackendArg] = {k: v for k, v in kwargs.items() if k not in ("equation", "dimension")}
        if "dimension" in kwargs:
            filtered_kwargs["axis"] = kwargs["dimension"]
        return ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())

    @classmethod
    def visit_Parameter(cls, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Visit parameter node.

        Args:
            node (LogicalNode): Parameter IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            str: Parameter comment assignment line.
        """
        del input_vars, kwargs
        return f"{node.id} = None # Parameter"

    @classmethod
    def visit_Return(cls, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Visit return node.

        Args:
            node (LogicalNode): Return IR node.
            input_vars (list[str]): Input variable names to return.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            str: Python return statement line.
        """
        del node, kwargs
        if not input_vars:
            return "return None"
        if len(input_vars) == 1:
            return f"return {input_vars[0]}"
        return "return " + ", ".join(input_vars)

    @classmethod
    def generic_visit(cls, node: BackendArg, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate default NumPy code.

        Args:
            node (BackendArg): The IR node.
            input_vars (list[str]): The names of the input variables.
            **kwargs (BackendArg): Additional keyword arguments.

        Returns:
            str: The generated default NumPy code string for the node.
        """
        op_type = str(adapt_backend_module(node).get_symbol("op_type", ""))
        if "dimension" in kwargs:
            return f"np.{op_type.lower()}({input_vars[0]}, axis={kwargs['dimension']})"
        op_spec = load_backend_mappings("numpy").operations.get(op_type)
        np_func = (op_spec.ast_template if op_spec and op_spec.ast_template else None) or f"np.{op_type.lower()}"
        args_str = ", ".join(input_vars)
        kwargs_str = cls._format_kwargs(kwargs)
        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str
        return f"{np_func}({args_str})"


@register_backend("numpy")
class NumpyGenerator(
    PythonStringGenerator,
):
    """Generate NumPy python code from IR."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the NumPy generator with an IR graph.

        Args:
            graph (IRGraph): The IR graph to generate code from.
        """
        super().__init__(graph)
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        schema = load_backend_mappings("numpy")
        self._import_header = "\n".join(schema.helpers or [])
        self.visitors.extend(
            [
                *get_shared_ast_visitors(generator=self),
                NumpyScatterVisitor(),
            ]
        )

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an operation eagerly using NumPy eager registry.

        Args:
            op_type (str): The operation type name.
            *args (BackendArg): Positional arguments for the operation.
            **kwargs (BackendArg): Keyword arguments for the operation.

        Returns:
            BackendArg: Computed result.
        """
        from ml_switcheroo_compiler.backends.numpy.eager import execute_op as numpy_execute_op

        return numpy_execute_op(cls, op_type, *args, **kwargs)

    @classmethod
    def get_module(cls) -> ModuleType:
        """Return the underlying NumPy module.

        Returns:
            ModuleType: The numpy module.
        """
        import sys

        import numpy as np

        _ = np
        return sys.modules["numpy"]

    @classmethod
    def get_scipy_module(cls, submodule: str = "special") -> ModuleType:
        """Return the requested SciPy submodule.

        Args:
            submodule (str): Submodule name ('special', 'linalg', 'stats').

        Returns:
            ModuleType: The requested scipy submodule.

        Raises:
            ValueError: If input arguments, shapes, or configuration values in `get_scipy_module` are invalid.
        """
        import importlib
        from collections.abc import Callable

        loaders: dict[str, Callable[[], ModuleType]] = {
            "special": lambda: importlib.import_module("scipy.special"),
            "linalg": lambda: importlib.import_module("scipy.linalg"),
            "stats": lambda: importlib.import_module("scipy.stats"),
            "signal": lambda: importlib.import_module("scipy.signal"),
            "fft": lambda: importlib.import_module("scipy.fft"),
            "ndimage": lambda: importlib.import_module("scipy.ndimage"),
            "sparse": lambda: importlib.import_module("scipy.sparse"),
            "optimize": lambda: importlib.import_module("scipy.optimize"),
            "integrate": lambda: importlib.import_module("scipy.integrate"),
        }
        loader = loaders.get(submodule)
        if loader is None:
            raise ValueError(f"Unsupported SciPy submodule: {submodule!r}")
        return loader()

    @classmethod
    def get_numpy_rng(cls, *args: BackendArg, **kwargs: BackendArg) -> TensorLike:
        """Get a numpy random generator.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns: np.ndarray: The rng.
        """
        import numpy as np

        return adapt_backend_module(np.random).require_symbol("default_rng")(*args, **kwargs)

    @classmethod
    def load(cls, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Load data using NumPy's load function.

        Args:
            *args (BackendArg): Positional arguments for np.load.
            **kwargs (BackendArg): Keyword arguments for np.load.

        Returns: np.ndarray: The loaded NumPy data.
        """
        import numpy as np

        return adapt_backend_module(np).require_op("load")(*args, **kwargs)

    @classmethod
    def save(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Save data using NumPy's save function.

        Args:
            *args (BackendArg): Positional arguments for np.save.
            **kwargs (BackendArg): Keyword arguments for np.save.: This function does not return a value.
        """
        import numpy as np

        adapt_backend_module(np).require_op("save")(*args, **kwargs)

    @classmethod
    def savez(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Save multiple arrays into a single file in uncompressed .npz format.

        Args:
            *args (BackendArg): Positional arguments for np.savez.
            **kwargs (BackendArg): Keyword arguments for np.savez.: This function does not return a value.
        """
        import numpy as np

        adapt_backend_module(np).require_op("savez")(*args, **kwargs)

    @classmethod
    def savez_compressed(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Save multiple arrays into a single file in compressed .npz format.

        Args:
            *args (BackendArg): Positional arguments for np.savez_compressed.
            **kwargs (BackendArg): Keyword arguments for np.savez_compressed.: This function does not return a value.
        """
        import numpy as np

        adapt_backend_module(np).require_op("savez_compressed")(*args, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Execute get helper functions for compiler processing.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        schema = load_backend_mappings("numpy")
        return schema.helpers or []

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate Python code for executing power iteration on a matrix.

        Args:
            node (IRNode): The IR node representing the PowerIteration operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (BackendArg): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for power iteration.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"np_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def get_fallback_prefix(self) -> str:
        """Retrieve the default fallback prefix string for NumPy operations from the generator.

        Returns:
            str: The fallback prefix string 'np'.
        """
        return NumpyTypeTranslator.get_fallback_prefix()

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into an executable callable wrapper for NumPy.

        Args:
            graph (IRGraph): The target computation graph.
            **kwargs (BackendArg): Compiler options.

        Returns:
            BackendArg: Callable execution wrapper.
        """
        import ml_switcheroo_compiler.interpreter.evaluator as evaluator_module
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        def callable_aot(*w_args: BackendArg, **w_kw: BackendArg) -> BackendArg:
            """Execute ahead-of-time compiled graph on inputs.

            Args:
                *w_args (BackendArg): Positional input tensors.
                **w_kw (BackendArg): Keyword arguments.

            Returns:
                BackendArg: Computed tensor or collection of tensors.
            """
            input_map = {}
            input_nodes = [
                n for n in graph.nodes.values() if adapt_backend_module(n).get_symbol("op_type", "") == "Input"
            ]
            for i, inp_node in enumerate(input_nodes):
                if i < len(w_args):
                    arg_val = w_args[i]
                    input_map[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val

            res_map = evaluator_module.evaluate_graph(graph, input_map)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    out_val: BackendArg = res_map[graph.outputs[0]]
                    if isinstance(out_val, Tensor):
                        return out_val
                    return Tensor(
                        out_val,
                        TensorConfig(
                            coerce_shape(adapt_backend_module(out_val).get_symbol("shape", ())), DType.Float32, "cpu"
                        ),
                    )
                outs: list[BackendArg] = []
                for out_id in graph.outputs:
                    out_val = res_map[out_id]
                    if isinstance(out_val, Tensor):
                        outs.append(out_val)
                    else:
                        outs.append(
                            Tensor(
                                out_val,
                                TensorConfig(
                                    coerce_shape(adapt_backend_module(out_val).get_symbol("shape", ())),
                                    DType.Float32,
                                    "cpu",
                                ),
                            )
                        )
                return outs
            return None

        return callable_aot

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for NumPy.

        Returns:
            EagerBackendAdapter: Concrete NumPy eager backend adapter.
        """
        import numpy as np

        from ml_switcheroo_compiler.backends.eager_registry import BackendOperand
        from ml_switcheroo_compiler.backends.protocols import StaticEagerBackendAdapter
        from ml_switcheroo_compiler.backends.typeclasses import (
            CoreEagerDictionary,
            CreationDictionary,
            ElementwiseBinaryDictionary,
            ElementwiseUnaryDictionary,
            LinearAlgebraDictionary,
            ReductionDictionary,
            verify_core_eager_typeclass,
        )

        np_adapter = adapt_backend_module(np, backend_name="numpy")
        tc_dict = verify_core_eager_typeclass(
            CoreEagerDictionary[BackendOperand](
                creation=CreationDictionary[BackendOperand](
                    zeros_fn=lambda shape, dtype: np_adapter.require_op("zeros")(shape, dtype=dtype),
                    ones_fn=lambda shape, dtype: np_adapter.require_op("ones")(shape, dtype=dtype),
                    array_fn=lambda data, dtype: np_adapter.require_op("array")(data, dtype=dtype),
                    asarray_fn=lambda data, dtype: np_adapter.require_op("asarray")(data, dtype=dtype),
                ),
                unary=ElementwiseUnaryDictionary[BackendOperand](
                    abs_fn=np_adapter.require_op("abs"),
                    exp_fn=np_adapter.require_op("exp"),
                    log_fn=np_adapter.require_op("log"),
                    sqrt_fn=np_adapter.require_op("sqrt"),
                    sin_fn=np_adapter.require_op("sin"),
                    cos_fn=np_adapter.require_op("cos"),
                    tanh_fn=np_adapter.require_op("tanh"),
                ),
                binary=ElementwiseBinaryDictionary[BackendOperand](
                    add_fn=np_adapter.require_op("add"),
                    subtract_fn=np_adapter.require_op("subtract"),
                    multiply_fn=np_adapter.require_op("multiply"),
                    divide_fn=np_adapter.require_op("divide"),
                ),
                linalg=LinearAlgebraDictionary[BackendOperand](
                    matmul_fn=np_adapter.require_op("matmul"),
                    reshape_fn=lambda x, shape: np_adapter.require_op("reshape")(x, shape),
                    transpose_fn=lambda x, axes: np_adapter.require_op("transpose")(x, axes),
                ),
                reduction=ReductionDictionary[BackendOperand](
                    sum_fn=lambda x, axis, keepdims: np_adapter.require_op("sum")(
                        x,
                        axis=axis,
                        keepdims=keepdims,
                    ),
                    mean_fn=lambda x, axis, keepdims: np_adapter.require_op("mean")(
                        x,
                        axis=axis,
                        keepdims=keepdims,
                    ),
                ),
            )
        )

        return StaticEagerBackendAdapter(
            backend_name="numpy",
            ops={
                "abs": np.abs,
                "add": np.add,
                "subtract": np.subtract,
                "multiply": np.multiply,
                "divide": np.divide,
                "sin": np.sin,
                "cos": np.cos,
                "tanh": np.tanh,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "matmul": np.matmul,
                "sum": np.sum,
                "mean": np.mean,
                "reshape": np.reshape,
                "transpose": np.transpose,
                "asarray": np.asarray,
                "array": np.array,
                "zeros": adapt_backend_module(np).require_symbol("zeros"),
                "ones": adapt_backend_module(np).require_symbol("ones"),
                "empty": adapt_backend_module(np).require_symbol("empty"),
                "copy": np.copy,
                "convolve": np.convolve,
                "correlate": np.correlate,
                "diagflat": np.diagflat,
                "issubdtype": np.issubdtype,
                "atleast_1d": np.atleast_1d,
                "nonzero": np.nonzero,
                "vstack": np.vstack,
                "hstack": np.hstack,
                "concatenate": np.concatenate,
                "ravel_multi_index": np.ravel_multi_index,
                "unravel_index": np.unravel_index,
                "argsort": np.argsort,
                "unique": np.unique,
                "intersect1d": np.intersect1d,
                "promote_types": np.promote_types,
                "where": np.where,
                "broadcast_to": np.broadcast_to,
                "outer": np.outer,
            },
            dtypes={
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
                "int8": np.int8,
                "int16": np.int16,
                "int32": np.int32,
                "int64": np.int64,
                "uint8": np.uint8,
                "bool": np.bool_,
                "bool_": np.bool_,
                "complex64": np.complex64,
                "complex128": np.complex128,
                "integer": np.integer,
                "number": np.number,
                "ndarray": np.ndarray,
                "inf": np.inf,
                "nan": np.nan,
            },
            submodules={
                "linalg": np.linalg,
                "fft": np.fft,
                "random": np.random,
            },
            module=np,
            typeclass_dict=tc_dict,
        )
