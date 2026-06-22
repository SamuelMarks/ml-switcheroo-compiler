"""Dask code generator and eager execution backend."""

try:
    import dask.array as da
except ImportError:
    da = None
from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


from ml_switcheroo_compiler.backends.common.generator_mixins import (
    SharedASTGeneratorMixin,
    GroupNormConfig,
)


@register_backend("dask")
class DaskGenerator(SharedASTGeneratorMixin, PythonStringGenerator):
    """Generates Dask python code from IR."""

    def _get_backend_prefix(self) -> str:
        return "da"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()
        res.extend(
            self._get_group_norm_code(
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
        return res

    _import_header = "import dask.array as da"
    _func_name = "evaluate"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"dask.einsum('{eq}', {args_str})"

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
            if args_str:
                args_str += f", {kwargs_str}"
            else:
                args_str = kwargs_str

        return f"{np_func}({args_str})"


if da is not None:  # pragma: no branch
    register_backend("dask")(DaskGenerator)
