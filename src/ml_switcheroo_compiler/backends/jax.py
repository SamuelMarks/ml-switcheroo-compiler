"""JAX/Flax Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("jax")
class JAXCodeGenerator(BaseGenerator):
    """Emit JAX-compatible pure functions from IR."""

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the JAX code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated JAX Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "jnp.matmul({0}, {1})",
            "Dot": "jnp.dot({0}, {1})",
            "BroadcastTo": "jnp.broadcast_to({0}, {shape})",
            "Reshape": "jnp.reshape({0}, {shape})",
            "TrueDivide": "jnp.true_divide({0}, {1})",
            "Zeros": "jnp.zeros({shape})",
            "Ones": "jnp.ones({shape})",
            "Full": "jnp.full({shape}, {fill_value})",
            "Arange": "jnp.arange({0})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
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
        return f"jnp.{op_type.lower()}({args_str})"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def generate(self) -> str:
        """Generate JAX code from the IR graph.

        Returns:
            str: The generated JAX Python code
        """
        self.code = [
            self.header.strip(),
            "import jax",
            "import jax.numpy as jnp\n",
        ]

        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)

    @classmethod
    def execute_op(cls, op_type: str, *args: object, **kwargs: object) -> object:
        """Execute op."""
        import jax.numpy

        try:
            func = getattr(jax.numpy, op_type.lower())
            return func(*args, **kwargs)
        except AttributeError:
            pass

        op_map = {
            "Add": getattr(jax.numpy, "add", None),
            "Subtract": getattr(jax.numpy, "subtract", None),
            "Multiply": getattr(jax.numpy, "multiply", None),
            "TrueDivide": getattr(jax.numpy, "divide", getattr(jax.numpy, "true_divide", None)),
            "Exp": getattr(jax.numpy, "exp", None),
            "Log": getattr(jax.numpy, "log", None),
            "Matmul": getattr(jax.numpy, "matmul", None),
            "Sin": getattr(jax.numpy, "sin", None),
            "Cos": getattr(jax.numpy, "cos", None),
            "Sum": getattr(jax.numpy, "sum", None),
            "Mean": getattr(jax.numpy, "mean", None),
            "Max": getattr(jax.numpy, "max", None),
            "Min": getattr(jax.numpy, "min", None),
            "Reshape": getattr(jax.numpy, "reshape", None),
            "Transpose": getattr(jax.numpy, "transpose", None),
            "Equal": getattr(jax.numpy, "equal", None),
            "NotEqual": getattr(jax.numpy, "not_equal", None),
            "Greater": getattr(jax.numpy, "greater", None),
            "Less": getattr(jax.numpy, "less", None),
            "Negative": getattr(jax.numpy, "negative", None),
        }

        if op_type in op_map and op_map[op_type] is not None:
            return op_map[op_type](*args, **kwargs)

        if op_type == "BroadcastTo":
            return jax.numpy.broadcast_to(*args, **kwargs)

        msg = f"Operation '{op_type}' is not supported by jax backend."
        raise NotImplementedError(msg)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> object:
        """Create zeros."""
        import jax.numpy

        return jax.numpy.zeros(shape)

    @classmethod
    def array(cls, data: object) -> object:
        """Create array."""
        import jax.numpy

        try:
            return jax.numpy.array(data)
        except AttributeError:
            return jax.numpy.convert_to_tensor(data)

    @classmethod
    def asarray(cls, data: object) -> object:
        """Convert array."""
        import jax.numpy

        try:
            return jax.numpy.asarray(data)
        except AttributeError:
            return jax.numpy.convert_to_tensor(data)

    @classmethod
    def item(cls, data: object) -> float:
        """Get item."""
        import jax.numpy

        try:
            return float(jax.numpy.asarray(data).item())
        except AttributeError:
            return float(data)
