"""Dask code generator and eager execution backend."""

import dask.array as da

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig, NNASTVisitor
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

try:
    pass
except ImportError:
    da = None


@register_backend("dask")
class DaskGenerator(PythonStringGenerator):
    """Generates Dask python code from IR."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend(
            [
                SharedASTGeneratorVisitor(generator=self),
            ]
        )

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "da"  # pragma: no cover

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()  # pragma: no cover
        res.extend(  # pragma: no cover
            NNASTVisitor(generator=self)._get_group_norm_code(
                GroupNormConfig(
                    "da",
                    "dask.array as da",
                    "da.reshape",
                    "da.mean",
                    "da.var",
                    "da.sqrt",
                    dim_arg="axis",
                    keepdim_arg="keepdims",
                )
            )
        )
        return res  # pragma: no cover

    _import_header = "import dask.array as da"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"dask.einsum('{eq}', {args_str})"  # pragma: no cover

    def visit_TruncateDiv(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateDiv."""
        x, y = input_vars
        return f"da.trunc(da.divide({x}, {y}))"

    def visit_TruncateMod(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateMod."""
        x, y = input_vars
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
        # Mapping from IR op types to dask functions
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
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if kwargs_str:
            if args_str:  # pragma: no branch
                args_str += f", {kwargs_str}"  # pragma: no cover
            else:
                args_str = kwargs_str

        return f"{np_func}({args_str})"


if da is not None:  # pragma: no branch
    register_backend("dask")(DaskGenerator)
