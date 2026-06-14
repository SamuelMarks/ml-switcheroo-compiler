"""CuPy code generator and eager execution backend."""

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None  # pragma: no cover

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

if cp is not None:  # pragma: no branch

    @register_backend("cupy")
    class CupyGenerator(BaseGenerator):
        """Generates CuPy python code from IR."""

        def generate(self) -> str:
            """Generate CuPy code."""
            self.code = [self.header]
            self.add_line("import cupy as cp")
            self.add_line("")
            self.add_line("def evaluate(args):")
            self.indent_level += 1

            self._generate_body("args")

            self.indent_level -= 1
            return "\n".join(self.code)

        def visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
            """Visit an IR node to emit code."""
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
                    args_str = kwargs_str  # pragma: no cover

            return f"{np_func}({args_str})"

        @classmethod
        def execute_op(cls, op_type: str, *args: object, **kwargs: object) -> object:
            """Eagerly execute an operation using CuPy."""
            op_map = {
                "Add": cp.add,
                "Subtract": cp.subtract,
                "Multiply": cp.multiply,
                "TrueDivide": cp.divide,
                "Exp": cp.exp,
                "Log": cp.log,
                "Matmul": cp.matmul,
                "Sin": cp.sin,
                "Cos": cp.cos,
                "Sum": cp.sum,
                "Mean": cp.mean,
                "Max": cp.max,
                "Min": cp.min,
                "BroadcastTo": cp.broadcast_to,
                "Reshape": cp.reshape,
                "Transpose": cp.transpose,
                "Equal": cp.equal,
                "NotEqual": cp.not_equal,
                "Greater": cp.greater,
                "Less": cp.less,
                "Negative": cp.negative,
            }

            if op_type in op_map:
                func = op_map[op_type]
            else:
                try:
                    func = getattr(cp, op_type.lower())
                except AttributeError:  # pragma: no cover
                    msg = f"Operation '{op_type}' not supported by cupy."  # pragma: no cover
                    raise NotImplementedError(msg) from None  # pragma: no cover

            return func(*args, **kwargs)

        @classmethod
        def zeros(cls, shape: tuple[int, ...]) -> object:
            """Create zeros."""
            return cp.zeros(shape)

        @classmethod
        def array(cls, data: object) -> object:
            """Create array."""
            return cp.array(data)

        @classmethod
        def asarray(cls, data: object) -> object:
            """Create asarray."""
            return cp.asarray(data)

        @classmethod
        def item(cls, data: object) -> float:
            """Get item."""
            return cp.asarray(data).item()
