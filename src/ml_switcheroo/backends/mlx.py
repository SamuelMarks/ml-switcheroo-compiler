"""MLX Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator


class MLXCodeGenerator(BaseGenerator):
    """Emit MLX-compatible code from IR."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return op_instance.emit_mlx(*args, **kwargs)

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Docstring."""
        self.add_line(f"{var_name} = mx.array({val_repr})")

    def generate(self) -> str:
        """Generate MLX model code from the IR graph.

        Returns:
            str: The generated MLX Python code.
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
