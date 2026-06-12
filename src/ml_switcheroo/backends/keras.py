"""Keras Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator
from ml_switcheroo.ir.core import IRNode


class KerasCodeGenerator(BaseGenerator):
    """Emit Keras Functional API script from IR."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Docstring."""
        super().__init__(*args, **kwargs)
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return op_instance.emit_keras(*args, **kwargs)

    def _emit_input_assignment(
        self, var_name: str, node: IRNode, input_prefix: str, input_idx: int
    ) -> None:
        """Docstring."""
        shape_str = (
            str(node.shape_metadata)
            if hasattr(node, "shape_metadata") and node.shape_metadata
            else "(None,)"
        )
        self.add_line(f"{var_name} = keras.Input(shape={shape_str}, name='{node.id}')")
        self.keras_input_vars.append(var_name)

    def _emit_output_assignment(
        self, node: IRNode, input_vars: list[str], returns: str
    ) -> None:
        """Docstring."""
        self.keras_output_vars.extend(input_vars)

    def generate(self) -> str:
        """Generate Keras model code from the IR graph.

        Returns:
            str: The generated Keras Python code.
        """
        self.code = [
            self.header.strip(),
            "import keras\n",
        ]

        self.indent_level = 0
        self.add_line("def get_model():")
        self.indent_level += 1

        self.keras_input_vars = []
        self.keras_output_vars = []

        self._generate_body()

        # Remove "return None" if it was added
        if self.code[-1].strip() == "return None":
            self.code.pop()

        inputs_str = ", ".join(self.keras_input_vars)
        outputs_str = ", ".join(self.keras_output_vars)
        self.add_line(
            f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])"
        )

        return "\n".join(self.code)
