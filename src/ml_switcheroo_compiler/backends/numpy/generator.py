# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator.py."""

from typing import Any

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

from .numpy_mixins import NumpyAudioVisitor, NumpyScatterVisitor, NumpyVisionVisitor


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
    def _format_kwargs(cls, kwargs: dict[str, Any]) -> str:
        """Evaluate _format_kwargs operation.

        Args:
        kwargs (object): The kwargs parameter.

        Returns:
        str: Result.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["equation", "dimension"]}
        if "dimension" in kwargs:
            filtered_kwargs["axis"] = kwargs["dimension"]
        return ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())

    @classmethod
    def visit_Parameter(cls, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Visit parameter."""
        return f"{node.id} = None # Parameter"

    @classmethod
    def visit_Return(cls, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Visit return."""
        if not input_vars:
            return "return None"
        if len(input_vars) == 1:
            return f"return {input_vars[0]}"
        return "return " + ", ".join(input_vars)

    @classmethod
    def visit_TriInv(cls, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Generate Python code for a triangular matrix inverse operation.

        Args:
            node (object): The IR node representing the TriInv operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for triangular matrix inverse.
        """
        return f"np.linalg.inv({input_vars[0]})"

    @classmethod
    def visit_TruncateDiv(cls, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate Python code for a truncated division operation.

        Args:
            node (IRNode): The IR node representing the TruncateDiv operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for truncated division.
        """
        x, y = input_vars
        return f"np.trunc(np.divide({x}, {y}))"

    @classmethod
    def visit_TruncateMod(cls, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate Python code for a truncated modulo operation.

        Args:
            node (IRNode): The IR node representing the TruncateMod operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for truncated modulo.
        """
        x, y = input_vars
        return f"np.fmod({x}, {y})"

    @classmethod
    def generic_visit(cls, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Generate default NumPy code.

        Args:
            node (object): The IR node.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated default NumPy code string for the node.
        """
        if "dimension" in kwargs:
            return f"np.{node.op_type.lower()}({input_vars[0]}, axis={kwargs['dimension']})"
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

    def __init__(self, graph: Any) -> None:
        """Initialize the NumPy generator with an IR graph.

        Args:
            graph (object): The IR graph to generate code from.
        """
        super().__init__(graph)
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        schema = load_backend_mappings("numpy")
        self._import_header = "\n".join(schema.helpers or [])
        self.visitors.extend(
            [
                *get_shared_ast_visitors(generator=self),
                NumpyVisionVisitor(),
                NumpyAudioVisitor(),
                NumpyScatterVisitor(),
            ]
        )

    @classmethod
    def get_numpy_rng(cls, *args: Any, **kwargs: Any) -> Any:
        """Get a numpy random generator.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: The rng.
        """
        import numpy as np

        return np.random.default_rng(*args, **kwargs)

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Any:
        """Load data using NumPy's load function.

        Args:
            *args (object): Positional arguments for np.load.
            **kwargs (object): Keyword arguments for np.load.

        Returns: Any: The loaded NumPy data.
        """
        import numpy as np

        return np.load(*args, **kwargs)

    @classmethod
    def save(cls, *args: Any, **kwargs: Any) -> None:
        """Save data using NumPy's save function.

        Args:
            *args (object): Positional arguments for np.save.
            **kwargs (object): Keyword arguments for np.save.: This function does not return a value.
        """
        import numpy as np

        np.save(*args, **kwargs)

    @classmethod
    def savez(cls, *args: Any, **kwargs: Any) -> None:
        """Save multiple arrays into a single file in uncompressed .npz format.

        Args:
            *args (object): Positional arguments for np.savez.
            **kwargs (object): Keyword arguments for np.savez.: This function does not return a value.
        """
        import numpy as np

        np.savez(*args, **kwargs)

    @classmethod
    def savez_compressed(cls, *args: Any, **kwargs: Any) -> None:
        """Save multiple arrays into a single file in compressed .npz format.

        Args:
            *args (object): Positional arguments for np.savez_compressed.
            **kwargs (object): Keyword arguments for np.savez_compressed.: This function does not return a value.
        """
        import numpy as np

        np.savez_compressed(*args, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Evaluate get_helper_functions operation.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        schema = load_backend_mappings("numpy")
        return schema.helpers or []

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate Python code for executing power iteration on a matrix.

        Args:
            node (IRNode): The IR node representing the PowerIteration operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

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
