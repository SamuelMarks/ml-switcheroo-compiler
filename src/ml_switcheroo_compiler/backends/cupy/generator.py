"""CuPy code generator and eager execution backend."""

try:
    import cupy as cp
except ImportError:
    cp = None
from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


from ml_switcheroo_compiler.backends.common.generator_mixins import (
    SharedASTGeneratorMixin,
    GroupNormConfig,
)


@register_backend("cupy")
class CupyGenerator(SharedASTGeneratorMixin, PythonStringGenerator):
    """Generates CuPy python code from IR."""

    def _get_backend_prefix(self) -> str:
        return "cp"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()
        res.extend(
            self._get_group_norm_code(
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
        return res

    _import_header = "import cupy as cp"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"cupy.einsum('{eq}', {args_str})"

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
            if args_str:
                args_str += f", {kwargs_str}"
            else:
                args_str = kwargs_str

        return f"{np_func}({args_str})"


if cp is not None:
    register_backend("cupy")(CupyGenerator)
