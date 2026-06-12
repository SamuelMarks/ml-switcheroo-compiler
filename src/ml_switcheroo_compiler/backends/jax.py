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
