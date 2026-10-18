"""Core abstractions and logic definitions for generator.py."""

from types import ModuleType
from typing import cast

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

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
    def _format_kwargs(cls, kwargs: EagerValue) -> str:
        """Execute format kwargs for compiler processing.

        Args:
        kwargs (EagerValue): The kwargs parameter.

        Returns:
        str: Result.
        """
        filtered_kwargs = {k: v for k, v in getattr(kwargs, "items")() if k not in getattr(["equation", "dimension"], "__iter__")()}
        if "dimension" in getattr(kwargs, "__iter__")():
            cast(EagerTensorProtocol, filtered_kwargs)["axis"] = getattr(kwargs, "__getitem__")("dimension")
        return ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())

    @classmethod
    def visit_Parameter(cls, node: LogicalNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Visit parameter.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            str: Computed result.
        """
        return f"{node.id} = None # Parameter"

    @classmethod
    def visit_Return(cls, node: LogicalNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Visit return.

        Args:
            node: The node parameter.
            input_vars: The input_vars parameter.
            **kwargs: The kwargs parameter.

        Returns:
            str: Computed result.
        """
        if not input_vars:
            return "return None"
        if len(input_vars) == 1:
            return f"return {input_vars[0]}"
        return "return " + ", ".join(input_vars)

    @classmethod
    def generic_visit(cls, node: EagerValue, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Generate default NumPy code.

        Args:
            node (EagerValue): The IR node.
            input_vars (list[str]): The names of the input variables.
            **kwargs (EagerValue): Additional keyword arguments.

        Returns:
            str: The generated default NumPy code string for the node.
        """
        if "dimension" in kwargs:
            return f"np.{getattr(node, 'op_type').lower()}({input_vars[0]}, axis={kwargs['dimension']})"
        op_type = getattr(node, "op_type", "")
        np_func = (lambda: __import__("ml_switcheroo_compiler.backends.mapping_loader", fromlist=["load_backend_mappings"]).load_backend_mappings("numpy").operations.get(op_type, type("Dummy", (), {"ast_template": None})()).ast_template or f"np.{op_type.lower()}")()
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
    def execute_op(cls, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Execute an operation eagerly using NumPy eager registry.

        Args:
            op_type (str): The operation type name.
            *args (EagerValue): Positional arguments for the operation.
            **kwargs (EagerValue): Keyword arguments for the operation.

        Returns:
            EagerValue: Computed result.
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
    def get_scipy_module(cls, submodule: str) -> ModuleType:
        """Return the requested SciPy submodule.

        Args:
            submodule (str): Submodule name ('special', 'linalg', 'stats').

        Returns:
            ModuleType: The requested scipy submodule.
        """
        import importlib

        return importlib.import_module(f"scipy.{submodule}")

    @classmethod
    def get_numpy_rng(cls, *args: EagerValue, **kwargs: EagerValue) -> TensorLike:
        """Get a numpy random generator.

        Args:
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns: np.ndarray: The rng.
        """
        import numpy as np

        return cast("TensorLike", getattr(np.random, "default_rng")(*args, **kwargs))

    @classmethod
    def load(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Load data using NumPy's load function.

        Args:
            *args (EagerValue): Positional arguments for np.load.
            **kwargs (EagerValue): Keyword arguments for np.load.

        Returns: np.ndarray: The loaded NumPy data.
        """
        import numpy as np

        return cast(EagerValue, getattr(np, "load")(*args, **kwargs))

    @classmethod
    def save(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Save data using NumPy's save function.

        Args:
            *args (EagerValue): Positional arguments for np.save.
            **kwargs (EagerValue): Keyword arguments for np.save.: This function does not return a value.
        """
        import numpy as np

        getattr(np, "save")(*args, **kwargs)

    @classmethod
    def savez(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Save multiple arrays into a single file in uncompressed .npz format.

        Args:
            *args (EagerValue): Positional arguments for np.savez.
            **kwargs (EagerValue): Keyword arguments for np.savez.: This function does not return a value.
        """
        import numpy as np

        getattr(np, "savez")(*args, **kwargs)

    @classmethod
    def savez_compressed(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Save multiple arrays into a single file in compressed .npz format.

        Args:
            *args (EagerValue): Positional arguments for np.savez_compressed.
            **kwargs (EagerValue): Keyword arguments for np.savez_compressed.: This function does not return a value.
        """
        import numpy as np

        getattr(np, "savez_compressed")(*args, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Execute get helper functions for compiler processing.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        schema = load_backend_mappings("numpy")
        return schema.helpers or []

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Generate Python code for executing power iteration on a matrix.

        Args:
            node (IRNode): The IR node representing the PowerIteration operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (EagerValue): Additional keyword arguments.

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

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an executable callable wrapper for NumPy.

        Args:
            graph (IRGraph): The target computation graph.
            **kwargs (EagerValue): Compiler options.

        Returns:
            EagerValue: Callable execution wrapper.
        """
        import ml_switcheroo_compiler.interpreter.evaluator as evaluator_module
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        def callable_aot(*w_args: EagerValue, **w_kw: EagerValue) -> EagerValue:
            """Execute ahead-of-time compiled graph on inputs.

            Args:
                *w_args (EagerValue): Positional input tensors.
                **w_kw (EagerValue): Keyword arguments.

            Returns:
                EagerValue: Computed tensor or collection of tensors.
            """
            input_map = {}
            input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
            for i, inp_node in enumerate(input_nodes):
                if i < len(w_args):
                    arg_val = w_args[i]
                    input_map[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val

            res_map = getattr(evaluator_module, "evaluate_graph")(graph, input_map)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    out_val = res_map[graph.outputs[0]]
                    if isinstance(out_val, Tensor):
                        return out_val
                    return Tensor(out_val, TensorConfig(getattr(out_val, "shape", ()), DType.Float32, "cpu"))
                outs = []
                for out_id in graph.outputs:
                    out_val = res_map[out_id]
                    if isinstance(out_val, Tensor):
                        outs.append(out_val)
                    else:
                        outs.append(Tensor(out_val, TensorConfig(getattr(out_val, "shape", ()), DType.Float32, "cpu")))
                return outs
            return None

        return callable_aot
