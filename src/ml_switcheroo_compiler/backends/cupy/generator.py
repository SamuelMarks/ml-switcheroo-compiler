"""CuPy code generator and eager execution backend."""

try:
    import cupy as cp
except ImportError:
    cp = None

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


class CupyGenerator(BaseGenerator):
    """Generates CuPy python code from IR."""

    def generate(self) -> str:
        """Execute generate.

        Returns:
        Any: The result.
        """
        self.code = [self.header]
        self.add_line("import cupy as cp")
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
