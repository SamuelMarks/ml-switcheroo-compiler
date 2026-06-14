"""Dask code generator and eager execution backend."""

try:
    import dask.array as da
except ImportError:
    da = None

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


class DaskGenerator(BaseGenerator):
    """Generates Dask python code from IR."""

    def generate(self) -> str:
        """Execute generate.

        Returns:
        Any: The result.
        """
        self.code = [self.header]
        self.add_line("import dask.array as da")
        self.add_line("")
        self.add_line("def evaluate(args):")
        self.indent_level += 1

        self._generate_body("args")

        self.indent_level -= 1
        return "\n".join(self.code)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Execute visit.

        Args:
            node (Any): Argument node.
            input_vars (Any): Argument input_vars.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
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
