"""MLX Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("mlx")
class MLXCodeGenerator(BaseGenerator):
    """Emit MLX-compatible code from IR."""

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the MLX code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated MLX Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "mx.matmul({0}, {1})",
            "Dot": "mx.dot({0}, {1})",
            "BroadcastTo": "mx.broadcast_to({0}, {shape})",
            "Reshape": "mx.reshape({0}, {shape})",
            "TrueDivide": "mx.divide({0}, {1})",
            "Zeros": "mx.zeros({shape})",
            "Ones": "mx.ones({shape})",
            "Full": "mx.full({shape}, {fill_value})",
            "Arange": "mx.arange({0})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": "mx.transpose({0}, {axes})" if "axes" in kwargs else "mx.transpose({0})",
            "Einsum": "mx.einsum({subscripts}, {0})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            # Replace kwargs placeholders
            for k, v in kwargs.items():
                if f"{{{k}}}" in fmt:
                    fmt = fmt.replace(f"{{{k}}}", str(v))
            # Replace args placeholders
            for i, var in enumerate(input_vars):
                fmt = fmt.replace(f"{{{i}}}", var)
            return fmt

        # Generic fallback
        args = list(input_vars)
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if kwargs.get("keepdims"):
            args.append(f"keepdims={kwargs['keepdims']}")

        args_str = ", ".join(args)
        return f"mx.{op_type.lower()}({args_str})"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = mx.array({val_repr})")

    def generate(self) -> str:
        """Generate MLX model code from the IR graph.

        Returns:
            str: The generated MLX Python code
        """
        self.code = [
            self.header.strip(),
            "import mlx.core as mx",
            "import mlx.nn as nn\n",
            "class CompiledModel(nn.Module):",
        ]
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")
        self.add_line("pass\n")
        self.indent_level -= 1

        self.add_line("def __call__(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)

    @classmethod
    def execute_op(cls, op_type: str, *args: object, **kwargs: object) -> object:
        """Execute op."""
        import mlx.core

        try:
            func = getattr(mlx.core, op_type.lower())
            return func(*args, **kwargs)
        except AttributeError:
            pass

        op_map = {
            "Add": getattr(mlx.core, "add", None),
            "Subtract": getattr(mlx.core, "subtract", None),
            "Multiply": getattr(mlx.core, "multiply", None),
            "TrueDivide": getattr(mlx.core, "divide", getattr(mlx.core, "true_divide", None)),
            "Exp": getattr(mlx.core, "exp", None),
            "Log": getattr(mlx.core, "log", None),
            "Matmul": getattr(mlx.core, "matmul", None),
            "Sin": getattr(mlx.core, "sin", None),
            "Cos": getattr(mlx.core, "cos", None),
            "Sum": getattr(mlx.core, "sum", None),
            "Mean": getattr(mlx.core, "mean", None),
            "Max": getattr(mlx.core, "max", None),
            "Min": getattr(mlx.core, "min", None),
            "Reshape": getattr(mlx.core, "reshape", None),
            "Transpose": getattr(mlx.core, "transpose", None),
            "Equal": getattr(mlx.core, "equal", None),
            "NotEqual": getattr(mlx.core, "not_equal", None),
            "Greater": getattr(mlx.core, "greater", None),
            "Less": getattr(mlx.core, "less", None),
            "Negative": getattr(mlx.core, "negative", None),
        }

        if op_type in op_map and op_map[op_type] is not None:
            return op_map[op_type](*args, **kwargs)

        if op_type == "BroadcastTo":
            return mlx.core.broadcast_to(*args, **kwargs)

        msg = f"Operation '{op_type}' is not supported by mlx backend."
        raise NotImplementedError(msg)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> object:
        """Create zeros."""
        import mlx.core

        return mlx.core.zeros(shape)

    @classmethod
    def array(cls, data: object) -> object:
        """Create array."""
        import mlx.core

        try:
            return mlx.core.array(data)
        except AttributeError:
            return mlx.core.convert_to_tensor(data)

    @classmethod
    def asarray(cls, data: object) -> object:
        """Convert array."""
        import mlx.core

        try:
            return mlx.core.asarray(data)
        except AttributeError:
            return mlx.core.convert_to_tensor(data)

    @classmethod
    def item(cls, data: object) -> float:
        """Get item."""
        import mlx.core

        try:
            return float(mlx.core.asarray(data).item())
        except AttributeError:
            return float(data)
