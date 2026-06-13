"""Keras Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("keras")
class KerasCodeGenerator(BaseGenerator):
    """Emit Keras Functional API script from IR."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes the object.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the Keras code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated Keras Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "keras.ops.matmul({0}, {1})",
            "Dot": "keras.ops.dot({0}, {1})",
            "BroadcastTo": "keras.ops.broadcast_to({0}, {shape})",
            "Reshape": "keras.ops.reshape({0}, {shape})",
            "TrueDivide": "keras.ops.true_divide({0}, {1})",
            "Zeros": "keras.ops.zeros({shape})",
            "Ones": "keras.ops.ones({shape})",
            "Full": "keras.ops.full({shape}, {fill_value})",
            "Arange": "keras.ops.arange({0})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": "keras.ops.transpose({0}, {axes})"
            if "axes" in kwargs
            else "keras.ops.transpose({0})",
            "Einsum": "keras.ops.einsum({subscripts}, {0})",
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
        return f"keras.ops.{op_type.lower()}({args_str})"

    def _emit_input_assignment(
        self,
        var_name: str,
        node: IRNode,
        input_prefix: str,
        input_idx: int,
    ) -> None:
        """Evaluate emit input assignment.

        Args:
            var_name (str): Argument var_name
            node (IRNode): Argument node
            input_prefix (str): Argument input_prefix
            input_idx (int): Argument input_idx
        """
        shape_str = (
            str(node.shape_metadata)
            if hasattr(node, "shape_metadata") and node.shape_metadata
            else "(None,)"
        )
        self.add_line(f"{var_name} = keras.Input(shape={shape_str}, name='{node.id}')")
        self.keras_input_vars.append(var_name)

    def _emit_output_assignment(
        self,
        node: IRNode,
        input_vars: list[str],
        returns: str,
    ) -> None:
        """Evaluate emit output assignment.

        Args:
            node (IRNode): Argument node
            input_vars (list[str]): Argument input_vars
            returns (str): Argument returns
        """
        self.keras_output_vars.extend(input_vars)

    def generate(self) -> str:
        """Generate Keras model code from the IR graph.

        Returns:
            str: The generated Keras Python code
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
            f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])",
        )

        return "\n".join(self.code)
