"""CuPy code generator and eager execution backend."""

import cupy as cp

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig, NNASTVisitor
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

try:
    pass
except ImportError:
    cp = None


@register_backend("cupy")
class CupyGenerator(PythonStringGenerator):
    """Generates CuPy python code from IR."""

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
        return "cp"  # pragma: no cover

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()  # pragma: no cover
        res.extend(  # pragma: no cover
            NNASTVisitor(generator=self)._get_group_norm_code(
                GroupNormConfig(
                    "cp",
                    "cupy as cp",
                    "cp.reshape",
                    "cp.mean",
                    "cp.var",
                    "cp.sqrt",
                    dim_arg="axis",
                    keepdim_arg="keepdims",
                )
            )
        )
        return res  # pragma: no cover

    _import_header = "import cupy as cp"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"cupy.einsum('{eq}', {args_str})"  # pragma: no cover

    def visit_TruncateDiv(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateDiv."""
        x, y = input_vars
        return f"cp.trunc(cp.divide({x}, {y}))"

    def visit_TruncateMod(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for TruncateMod."""
        x, y = input_vars
        return f"cp.fmod({x}, {y})"

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
        # Mapping from IR op types to cupy functions
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
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if kwargs_str:
            if args_str:  # pragma: no branch
                args_str += f", {kwargs_str}"  # pragma: no cover
            else:
                args_str = kwargs_str

        return f"{np_func}({args_str})"


if cp is not None:
    register_backend("cupy")(CupyGenerator)
