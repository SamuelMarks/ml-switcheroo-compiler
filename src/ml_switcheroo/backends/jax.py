"""JAX/Flax Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator


class JAXCodeGenerator(BaseGenerator):
    """Emit JAX-compatible pure functions from IR."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return op_instance.emit_jax(*args, **kwargs)

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Docstring."""
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def generate(self) -> str:
        """Generate JAX code from the IR graph.

        Returns:
            str: The generated JAX Python code.
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
