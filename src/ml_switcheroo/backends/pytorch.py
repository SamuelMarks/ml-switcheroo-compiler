"""PyTorch Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator


class PyTorchCodeGenerator(BaseGenerator):
    """Emit PyTorch-compatible code from IR."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return op_instance.emit_pytorch(*args, **kwargs)

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Docstring."""
        self.add_line(f"{var_name} = self.{var_name}")

    def generate(self) -> str:
        """Generate PyTorch model code from the IR graph.

        Returns:
            str: The generated PyTorch Python code.
        """
        self.code = [
            self.header.strip(),
            "import torch",
            "import torch.nn as nn\n",
            "class CompiledModel(nn.Module):",
        ]

        # __init__
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")

        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(
                    f"self.register_parameter('{var_name}', "
                    f"nn.Parameter(torch.tensor({val_repr})))"
                )
                has_params = True

        if not has_params:
            self.add_line("pass")

        self.add_line("")
        self.indent_level -= 1

        # forward
        self.add_line("def forward(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
