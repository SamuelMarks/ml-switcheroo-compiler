"""TensorFlow Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator


class TensorFlowCodeGenerator(BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return op_instance.emit_tensorflow(*args, **kwargs)

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Docstring."""
        self.add_line(f"{var_name} = tf.constant({val_repr})")

    def generate(self) -> str:
        """Generate TensorFlow model code from the IR graph.

        Returns:
            str: The generated TensorFlow Python code.
        """
        self.code = [
            self.header.strip(),
            "import tensorflow as tf\n",
        ]

        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
