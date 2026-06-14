"""NumPy code generator and eager execution backend."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("numpy")
class NumpyGenerator(BaseGenerator):
    """Generates NumPy python code from IR."""

    def generate(self) -> str:
        """Execute generate.

        Returns:
        Any: The result.
        """
        self.code = [self.header]
        self.add_line("import numpy as np")
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
        # Mapping from IR op types to numpy functions
        op_map = {
            "Add": "np.add",
            "Subtract": "np.subtract",
            "Multiply": "np.multiply",
            "TrueDivide": "np.divide",
            "Exp": "np.exp",
            "Log": "np.log",
            "Matmul": "np.matmul",
            "Sin": "np.sin",
            "Cos": "np.cos",
            "Sum": "np.sum",
            "Mean": "np.mean",
            "Max": "np.max",
            "Min": "np.min",
            "BroadcastTo": "np.broadcast_to",
            "Reshape": "np.reshape",
            "Transpose": "np.transpose",
            "Equal": "np.equal",
            "NotEqual": "np.not_equal",
            "Greater": "np.greater",
            "Less": "np.less",
            "Negative": "np.negative",
        }

        np_func = op_map.get(op_type, f"np.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

        return f"{np_func}({args_str})"
