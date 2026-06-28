"""TensorFlow Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("tensorflow")
class TensorFlowCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "tf"  # pragma: no cover

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        """Function docstring.

        Args:
        op: Arg.
        kwargs: Arg.
        """
        res = f"tf.{op}({{shape}})"
        if "dtype" in kwargs:  # pragma: no branch
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    def _format_full(self, kwargs: object) -> str:
        """Function docstring.

        Args:
        kwargs: Arg.
        """
        res = "tf.full({shape}, {fill_value})"
        if "dtype" in kwargs:  # pragma: no branch
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    def _format_transpose(self, kwargs: object) -> str:
        """Function docstring.

        Args:
        kwargs: Arg.
        """
        if "axes" in kwargs:  # pragma: no branch
            return "tf.transpose({0}, perm={axes})"  # pragma: no cover
        return "tf.transpose({0})"

    def visit_ConvTranspose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ConvTranspose."""
        lhs = input_vars[0]
        rhs = input_vars[1]
        strides = node.attributes.get("strides", 1)
        padding = node.attributes.get("padding", "VALID")
        return f"tf_conv_transpose({lhs}, {rhs}, {strides}, '{padding}')"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"tf_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Extra arguments.

        Returns:
            str: The code string.
        """
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"tf.einsum('{eq}', {args_str})"  # pragma: no cover

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "tf.math"

    def _get_math_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "TruncateDiv": "tf.math.truncatediv({0}, {1})",
            "TruncateMod": "tf.math.truncatemod({0}, {1})",
            "TrueDivide": "tf.math.truediv({0}, {1})",
            "Sum": "tf.reduce_sum({0}, axis={axis}, keepdims={keepdims})",
            "Mean": "tf.reduce_mean({0}, axis={axis}, keepdims={keepdims})",
            "Max": "tf.reduce_max({0}, axis={axis}, keepdims={keepdims})",
            "Min": "tf.reduce_min({0}, axis={axis}, keepdims={keepdims})",
            "Prod": "tf.reduce_prod({0}, axis={axis}, keepdims={keepdims})",
            "All": "tf.reduce_all({0}, axis={axis}, keepdims={keepdims})",
            "AnyOp": "tf.reduce_any({0}, axis={axis}, keepdims={keepdims})",
            "Erfinv": "tf.math.erfinv({0})",
            "NanToNum": "tf.where(tf.math.is_nan({0}), {nan}, tf.where(tf.math.is_inf({0}) & ({0} > 0), {posinf}, tf.where(tf.math.is_inf({0}) & ({0} < 0), {neginf}, {0})))",
        }

    def _get_linalg_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "Matmul": "tf.linalg.matmul({0}, {1})",
            "Trace": "tf.linalg.trace",
            "Adjoint": "tf.linalg.adjoint",
            "BandPart": "tf.linalg.band_part",
            "CholeskySolve": "tf.linalg.cholesky_solve",
            "TriInv": "tf.linalg.inv({0})",
            "BandedTriangularSolve": "tf.linalg.banded_triangular_solve",
            "EighTridiagonal": "tf.linalg.eigh_tridiagonal",
            "MatrixRank": "tf.linalg.matrix_rank",
            "MatrixTranspose": "tf.linalg.matrix_transpose",
            "Sqrtm": "tf.linalg.sqrtm",
            "Dot": "tf.tensordot({0}, {1}, axes=1)",
            "Fftnd": "tf.signal.fftn({0})",
            "Ifftnd": "tf.signal.ifftn({0})",
            "Rfftnd": "tf.signal.rfftn({0})",
            "Irfftnd": "tf.signal.irfftn({0})",
            "Fftshift": "tf.signal.fftshift({0})",
            "Ifftshift": "tf.signal.ifftshift({0})",
            "Dct": "tf.signal.dct({0})",
            "Idct": "tf.signal.idct({0})",
            "Mdct": "tf.signal.mdct({0})",
            "InverseMdct": "tf.signal.inverse_mdct({0})",
            "Frame": "tf.signal.frame({0})",
            "OverlapAndAdd": "tf.signal.overlap_and_add({0})",
            "Fft": "tf.signal.fft({0})",
            "Rfft": "tf.signal.rfft({0})",
            "Fftn": "tf.signal.fftNd({0})",
        }

    def _get_nn_ops(self, kwargs: dict) -> dict[str, str]:
        return {
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

    def _get_creation_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "Arange": "tf.range({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
        }

    def _get_array_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "BroadcastTo": "tf.broadcast_to({0}, {shape})",
            "Reshape": "tf.reshape({0}, {shape})",
            "Sort": "tf.sort({0}, axis={dimension})",
            "ArgSort": "tf.argsort({0}, axis={dimension})",
            "Allclose": "tf.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "AssignVariable": "{0}",
            "StopGradient": "tf.stop_gradient({0})",
            "Resize": "tf.image.resize({0}, {size}, method={method}, antialias={antialias})",
            "AffineGrid": "tf.raw_ops.AffineGrid(theta={0}, size={size}, align_corners={align_corners})",
            "GridSample": "tf.raw_ops.GridSample(input={0}, grid={1}, mode={mode}, padding_mode={padding_mode}, align_corners={align_corners})",
            "DrawBoundingBoxes": "tf.image.draw_bounding_boxes({0}, {1}, colors={colors}, texts={texts})",
            "RgbToYiq": "tf.image.rgb_to_yiq({0})",
            "YiqToRgb": "tf.image.yiq_to_rgb({0})",
            "RgbToYuv": "tf.image.rgb_to_yuv({0})",
            "YuvToRgb": "tf.image.yuv_to_rgb({0})",
            "Ifft": "tf.signal.ifft({0})",
            "Fft2d": "tf.signal.fft2d({0})",
            "Ifft2d": "tf.signal.ifft2d({0})",
            "Fft3d": "tf.signal.fft3d({0})",
            "Ifft3d": "tf.signal.ifft3d({0})",
            "Rfft2d": "tf.signal.rfft2d({0})",
            "Rfft3d": "tf.signal.rfft3d({0})",
            "Irfft": "tf.signal.irfft({0})",
            "Irfft2d": "tf.signal.irfft2d({0})",
            "Irfft3d": "tf.signal.irfft3d({0})",
            "Stft": "tf.signal.stft({0}, frame_length={win_length}, frame_step={hop_length}, fft_length={n_fft}, window_fn={window}, pad_end={center})",
            "Istft": "tf.signal.inverse_stft({0}, frame_length={win_length}, frame_step={hop_length}, fft_length={n_fft}, window_fn={window})",
            "HannWindow": "tf.signal.hann_window({window_length}, periodic={periodic})",
            "HammingWindow": "tf.signal.hamming_window({window_length}, periodic={periodic})",
            "MfccsFromLogMelSpectrograms": "tf.signal.mfccs_from_log_mel_spectrograms({0})[..., :{num_mfccs}]",
            "ReadVariable": "{0}",
            "Transpose": self._format_transpose(kwargs),
            "Argmax": "tf.math.argmax({0}, axis={axis})",
            "Argmin": "tf.math.argmin({0}, axis={axis})",
            "Cast": "tf.cast({0}, dtype=tf.{dtype})",
            "Bitcast": "tf.bitcast({0}, type=tf.{dtype})",
        }

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        ops = {}
        ops.update(self._get_math_ops(kwargs))
        ops.update(self._get_linalg_ops(kwargs))
        ops.update(self._get_nn_ops(kwargs))
        ops.update(self._get_creation_ops(kwargs))
        ops.update(self._get_array_ops(kwargs))

        ops["Beta"] = (
            "tf.random.gamma({shape}, alpha={1}) / (tf.random.gamma({shape}, alpha={1}) + tf.random.gamma({shape}, alpha={2}))"
        )
        ops["Dirichlet"] = (
            "tf.random.gamma({shape}, alpha={1}) / tf.reduce_sum(tf.random.gamma({shape}, alpha={1}), axis=-1, keepdims=True)"
        )
        ops["Gamma"] = "tf.random.gamma({shape}, alpha={1})"
        ops["RngBitGenerator"] = "tf.random.uniform({shape}, minval=0, maxval=255, dtype=tf.int32)"
        ops["RngUniform"] = "tf.random.uniform({shape}, minval={0}, maxval={1})"

        ops["Infeed"] = "{0}"
        ops["Outfeed"] = "{0}"
        ops["AxisIndex"] = "0"
        ops["WithShardingConstraint"] = "{0}"
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = tf.constant({val_repr})")

    def _generate_file_header(self) -> list[str]:
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        from ml_switcheroo_compiler.backends.common.generator_mixins import GroupNormConfig

        return [
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
        ]

    def _generate_function_signature(self) -> None:
        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs):")
        self.indent_level += 1
