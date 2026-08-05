# ruff: noqa: E501
"""CuPy code generator and eager execution backend."""

try:
    import cupy as cp
except ImportError:
    cp = None

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("cupy")
class CupyGenerator(PythonStringGenerator):
    """Generate CuPy python code from IR."""

    def __init__(self, graph: object) -> None:
        """Init.

        Args:
            graph (object): The graph parameter.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "cp"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions.

        Returns:
        object: Result.
        """
        res = []
        return res

    _import_header = "import cupy as cp"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes.

        Args:
            node (IRNode): The node parameter.
            input_vars (list): The input_vars parameter.
            **kwargs (object): Keyword args.

        Returns:
            str: Result.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"cupy.einsum('{eq}', {args_str})"

    def visit_TruncateDiv(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateDiv.

        Args:
            node (IRNode): The node parameter.
            input_vars (list): The input_vars parameter.
            **kwargs (object): Keyword args.

        Returns:
            str: Result.
        """
        (x, y) = input_vars
        return f"cp.trunc(cp.divide({x}, {y}))"

    def visit_TruncateMod(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateMod.

        Args:
            node (IRNode): The node parameter.
            input_vars (list): The input_vars parameter.
            **kwargs (object): Keyword args.

        Returns:
            str: Result.
        """
        (x, y) = input_vars
        return f"cp.fmod({x}, {y})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): The input_vars parameter.
            **kwargs: Extra attributes.

        Returns:
            str: Generated code.
        """
        op_type = node.op_type
        op_map = {
            "Add": "cp.add",
            "Subtract": "cp.subtract",
            "Multiply": "cp.multiply",
            "TrueDivide": "cp.divide",
            "Exp": "cp.exp",
            "Log": "cp.log",
            "Matmul": "cp.matmul",
            "Sin": "cp.sin",
            "Acos": "cp.arccos",
            "Acosh": "cp.arccosh",
            "Asin": "cp.arcsin",
            "Asinh": "cp.arcsinh",
            "Atan": "cp.arctan",
            "Atan2": "cp.arctan2",
            "Atanh": "cp.arctanh",
            "Cos": "cp.cos",
            "Sum": "cp.sum",
            "Mean": "cp.mean",
            "Max": "cp.max",
            "Min": "cp.min",
            "BroadcastTo": "cp.broadcast_to",
            "Reshape": "cp.reshape",
            "Transpose": "cp.transpose",
            "Equal": "cp.equal",
            "NotEqual": "cp.not_equal",
            "Greater": "cp.greater",
            "Less": "cp.less",
            "Negative": "cp.negative",
        }
        np_func = op_map.get(op_type, f"cp.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = ", ".join(f"{k}={v}" for (k, v) in kwargs.items())
        if kwargs_str:
            if args_str:
                args_str += f", {kwargs_str}"
            else:
                args_str = kwargs_str
        return f"{np_func}({args_str})"


if cp is not None:
    register_backend("cupy")(CupyGenerator)
