"""TensorFlow Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("tensorflow")
class TensorFlowCodeGenerator(BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the TensorFlow code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated TensorFlow Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "tf.linalg.matmul({0}, {1})",
            "Dot": "tf.tensordot({0}, {1}, axes=1)",
            "BroadcastTo": "tf.broadcast_to({0}, {shape})",
            "Reshape": "tf.reshape({0}, {shape})",
            "TrueDivide": "tf.math.truediv({0}, {1})",
            "Zeros": "tf.zeros({shape})",
            "Ones": "tf.ones({shape})",
            "Full": "tf.fill({shape}, {fill_value})",
            "Arange": "tf.range({0})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": "tf.transpose({0}, perm={axes})"
            if "axes" in kwargs
            else "tf.transpose({0})",
            "Einsum": "tf.einsum({subscripts}, {0})",
            "Sum": "tf.reduce_sum({0}, axis={axis}, keepdims={keepdims})",
            "Mean": "tf.reduce_mean({0}, axis={axis}, keepdims={keepdims})",
            "Max": "tf.reduce_max({0}, axis={axis}, keepdims={keepdims})",
            "Min": "tf.reduce_min({0}, axis={axis}, keepdims={keepdims})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            # Replace kwargs placeholders
            for k, v in kwargs.items():
                if f"{{{k}}}" in fmt:
                    fmt = fmt.replace(f"{{{k}}}", str(v))
            # Special cases for axis/keepdims
            if "keepdims" in fmt and "keepdims" not in kwargs:
                fmt = fmt.replace(", keepdims={keepdims}", "")
            if "axis" in fmt and "axis" not in kwargs:
                fmt = fmt.replace(", axis={axis}", "")
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
        return f"tf.math.{op_type.lower()}({args_str})"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = tf.constant({val_repr})")

    def generate(self) -> str:
        """Generate TensorFlow model code from the IR graph.

        Returns:
            str: The generated TensorFlow Python code
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
