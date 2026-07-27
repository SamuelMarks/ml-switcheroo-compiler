# ruff: noqa: E501
"""Dask code generator and eager execution backend."""

try:
    import dask.array as da
except ImportError:
    da = None

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("dask")
class DaskGenerator(PythonStringGenerator):
    """Generates Dask python code from IR."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def _get_backend_prefix(self) -> str:
        r"""Get the library prefix string used when emitting Dask array operations.\n\n        Returns:\n            str: The string \'da\'.\n."""
        return "da"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = []
        return res

    _import_header = "import dask.array as da"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"dask.einsum('{eq}', {args_str})"

    def visit_TruncateDiv(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateDiv."""
        (x, y) = input_vars
        return f"da.trunc(da.divide({x}, {y}))"

    def visit_TruncateMod(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateMod."""
        (x, y) = input_vars
        return f"da.fmod({x}, {y})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (IRNode): Argument node.
            input_vars (list[str]): Argument input_vars.
            **kwargs: Extra attributes.

        Returns:
            str: Generated code.
        """
        op_type = node.op_type
        op_map = {
            "Add": "da.add",
            "Subtract": "da.subtract",
            "Multiply": "da.multiply",
            "TrueDivide": "da.divide",
            "Exp": "da.exp",
            "Log": "da.log",
            "Matmul": "da.matmul",
            "Sin": "da.sin",
            "Acos": "da.arccos",
            "Acosh": "da.arccosh",
            "Asin": "da.arcsin",
            "Asinh": "da.arcsinh",
            "Atan": "da.arctan",
            "Atan2": "da.arctan2",
            "Atanh": "da.arctanh",
            "Cos": "da.cos",
            "Sum": "da.sum",
            "Mean": "da.mean",
            "Max": "da.max",
            "Min": "da.min",
            "BroadcastTo": "da.broadcast_to",
            "Reshape": "da.reshape",
            "Transpose": "da.transpose",
            "Equal": "da.equal",
            "NotEqual": "da.not_equal",
            "Greater": "da.greater",
            "Less": "da.less",
            "Negative": "da.negative",
        }
        np_func = op_map.get(op_type, f"da.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = ", ".join(f"{k}={v}" for (k, v) in kwargs.items())
        if kwargs_str:
            if args_str:
                args_str += f", {kwargs_str}"
            else:
                args_str = kwargs_str
        return f"{np_func}({args_str})"


if da is not None:
    register_backend("dask")(DaskGenerator)
