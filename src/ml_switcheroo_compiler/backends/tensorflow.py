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
            "Prod": "tf.reduce_prod({0}, axis={axis}, keepdims={keepdims})",
            "All": "tf.reduce_all({0}, axis={axis}, keepdims={keepdims})",
            "AnyOp": "tf.reduce_any({0}, axis={axis}, keepdims={keepdims})",
            "Argmax": "tf.math.argmax({0}, axis={axis})",
            "Argmin": "tf.math.argmin({0}, axis={axis})",
            "Cast": "tf.cast({0}, dtype=tf.{dtype})",
            "Bitcast": "tf.bitcast({0}, type=tf.{dtype})",
            "Relu": "tf.nn.relu({0})",
            "Relu6": "tf.nn.relu6({0})",
            "LeakyRelu": "tf.nn.leaky_relu({0}, alpha={alpha})",
            "Elu": "tf.nn.elu({0})",
            "Selu": "tf.nn.selu({0})",
            "Gelu": "tf.nn.gelu({0}, approximate={approximate})",
            "Sigmoid": "tf.math.sigmoid({0})",
            "Softmax": "tf.nn.softmax({0}, axis={axis})",
            "LogSoftmax": "tf.nn.log_softmax({0}, axis={axis})",
            "Softplus": "tf.math.softplus({0})",
            "Softsign": "tf.math.softsign({0})",
            "Conv1D": "tf.nn.conv1d({0}, {1}, stride={stride}, padding={padding})",
            "Conv2D": "tf.nn.conv2d({0}, {1}, strides={strides}, padding={padding})",
            "Conv3D": "tf.nn.conv3d({0}, {1}, strides={strides}, padding={padding})",
            "MaxPool1D": "tf.nn.max_pool1d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
            "MaxPool2D": "tf.nn.max_pool2d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
            "MaxPool3D": "tf.nn.max_pool3d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
            "AvgPool1D": "tf.nn.avg_pool1d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
            "AvgPool2D": "tf.nn.avg_pool2d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
            "AvgPool3D": "tf.nn.avg_pool3d({0}, ksize={ksize}, "
            "strides={strides}, padding={padding})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            # Replace kwargs placeholders
            for k, v in kwargs.items():
                if f"{{{k}}}" in fmt:
                    fmt = fmt.replace(f"{{{k}}}", str(v))
            # Strip remaining unmatched kwargs in the form `, key={key}`
            import re

            fmt = re.sub(r", \w+=\{[^\}]+\}", "", fmt)

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

    @classmethod
    def execute_op(cls, op_type: str, *args: object, **kwargs: object) -> object:
        """Execute op."""
        import tensorflow as tf

        try:
            func = getattr(tf.math, op_type.lower())
            return func(*args, **kwargs)  # pragma: no cover
        except AttributeError:
            pass

        op_map = {
            "Add": getattr(tf.math, "add", None),
            "Subtract": getattr(tf.math, "subtract", None),
            "Multiply": getattr(tf.math, "multiply", None),
            "TrueDivide": getattr(tf.math, "divide", getattr(tf.math, "true_divide", None)),
            "Exp": getattr(tf.math, "exp", None),
            "Log": getattr(tf.math, "log", None),
            "Matmul": getattr(tf.math, "matmul", None),
            "Sin": getattr(tf.math, "sin", None),
            "Cos": getattr(tf.math, "cos", None),
            "Sum": getattr(tf.math, "sum", None),
            "Mean": getattr(tf.math, "mean", None),
            "Max": getattr(tf.math, "max", None),
            "Min": getattr(tf.math, "min", None),
            "Reshape": getattr(tf.math, "reshape", None),
            "Transpose": getattr(tf.math, "transpose", None),
            "Equal": getattr(tf.math, "equal", None),
            "NotEqual": getattr(tf.math, "not_equal", None),
            "Greater": getattr(tf.math, "greater", None),
            "Less": getattr(tf.math, "less", None),
            "Negative": getattr(tf.math, "negative", None),
        }

        if op_type in op_map and op_map[op_type] is not None:  # pragma: no cover
            return op_map[op_type](*args, **kwargs)  # pragma: no cover
        # pragma: no cover
        if op_type == "BroadcastTo":  # pragma: no cover
            return tf.math.broadcast_to(*args, **kwargs)  # pragma: no cover
        # pragma: no cover
        msg = f"Operation '{op_type}' is not supported by tensorflow backend."  # pragma: no cover
        raise NotImplementedError(msg)  # pragma: no cover

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> object:
        """Create zeros."""
        import tensorflow as tf

        return tf.zeros(shape)

    @classmethod
    def array(cls, data: object) -> object:
        """Create array."""
        import tensorflow as tf

        try:
            return tf.math.array(data)
        except AttributeError:
            return tf.convert_to_tensor(data)

    @classmethod
    def asarray(cls, data: object) -> object:
        """Convert array."""
        import tensorflow as tf

        try:
            return tf.math.asarray(data)
        except AttributeError:
            return tf.convert_to_tensor(data)

    @classmethod
    def item(cls, data: object) -> float:
        """Get item."""
        import tensorflow as tf

        try:
            return float(tf.math.asarray(data).item())
        except AttributeError:
            return float(data)
