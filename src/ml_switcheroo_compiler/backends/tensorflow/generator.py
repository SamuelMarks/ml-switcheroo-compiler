"""TensorFlow Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("tensorflow")
class TensorFlowCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    def _get_backend_prefix(self) -> str:
        return "tf"

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        res = f"tf.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        res = "tf.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs: object) -> str:
        if "axes" in kwargs:
            return "tf.transpose({0}, perm={axes})"
        return "tf.transpose({0})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Extra arguments.

        Returns:
            str: The code string.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"tf.einsum('{eq}', {args_str})"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback visit method for generic nodes.

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
            "Arange": "tf.range({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "tf.sort({0}, axis={dimension})",
            "ArgSort": "tf.argsort({0}, axis={dimension})",
            "Allclose": "tf.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "tf.signal.fft({0})",
            "Rfft": "tf.signal.rfft({0})",
            "Fftn": "tf.signal.fftNd({0})",
            "Erfinv": "tf.math.erfinv({0})",
            "NanToNum": "tf.where(tf.math.is_nan({0}), {nan}, tf.where(tf.math.is_inf({0}) & ({0} > 0), {posinf}, tf.where(tf.math.is_inf({0}) & ({0} < 0), {neginf}, {0})))",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": self._format_transpose(kwargs),
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
            "MaxPool1D": "tf.nn.max_pool1d({0}, ksize={ksize}, strides={strides}, padding={padding})",
            "MaxPool2D": "tf.nn.max_pool2d({0}, ksize={ksize}, strides={strides}, padding={padding})",
            "MaxPool3D": "tf.nn.max_pool3d({0}, ksize={ksize}, strides={strides}, padding={padding})",
            "AvgPool1D": "tf.nn.avg_pool1d({0}, ksize={ksize}, strides={strides}, padding={padding})",
            "AvgPool2D": "tf.nn.avg_pool2d({0}, ksize={ksize}, strides={strides}, padding={padding})",
            "AvgPool3D": "tf.nn.avg_pool3d({0}, ksize={ksize}, strides={strides}, padding={padding})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            fmt = OpFormatter.format_backend_string(fmt, input_vars, kwargs)
            import re

            fmt = re.sub(r", \w+=\{[^\}]+\}", "", fmt)
            return fmt

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(
                prefix="tf.math", op_type=op_type, input_vars=input_vars, kwargs=kwargs
            )
        )

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
        from ml_switcheroo_compiler.backends.common.generator_mixins import GroupNormConfig

        self.code = [
            self.header.strip(),
            "import tensorflow as tf\n",
            *self._get_group_norm_code(
                GroupNormConfig(
                    prefix="tf",
                    module="tensorflow as tf",
                    reshape="tf.reshape",
                    mean="tf.reduce_mean",
                    var="tf.math.reduce_variance",
                    sqrt="tf.math.sqrt",
                    dim_arg="axis",
                    keepdim_arg="keepdims",
                )
            ),
            "",
        ]

        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
