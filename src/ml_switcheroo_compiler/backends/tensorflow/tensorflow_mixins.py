# ruff: noqa: E501
"""Mixins."""


class TensorFlowMathMixin:
    """Math Mixin."""

    def _get_math_ops(self, kwargs: dict) -> dict[str, str]:
        """Evaluate _get_math_ops operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
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
        """Evaluate _get_linalg_ops operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
        return {
            "Matmul": "tf.linalg.matmul({0}, {1})",
            "Trace": "tf.linalg.trace",
            "Adjoint": "tf.linalg.adjoint",
            "LuMatrixInverse": "tf.linalg.lu_matrix_inverse({0}, {1})",
            "LuReconstruct": "tf.linalg.lu_reconstruct({0}, {1})",
            "BandPart": "tf.linalg.band_part",
            "TriangularSolve": "tf.linalg.triangular_solve({0}, {1}, lower={lower}, adjoint={adjoint})",
            "TridiagonalSolve": "tf.linalg.tridiagonal_solve(({2}, {1}, {0}), {3}, diagonals_format='sequence')",
            "TridiagonalMatmul": "tf.linalg.tridiagonal_matmul(({2}, {1}, {0}), {3}, diagonals_format='sequence')",
            "CholeskySolve": "tf.linalg.cholesky_solve({0}, {1})",
            "TriInv": "tf.linalg.inv({0})",
            "Dot": "tf.tensordot({0}, {1}, axes=1)",
            "Fftnd": "tf.signal.fftn({0})",
            "Ifftnd": "tf.signal.ifftn({0})",
            "Rfftnd": "tf.signal.rfftn({0})",
            "Irfftnd": "tf.signal.irfftn({0})",
            "Fftshift": "tf.signal.fftshift({0})",
            "Ifftshift": "tf.signal.ifftshift({0})",
            "Fft": "tf.signal.fft({0})",
            "Rfft": "tf.signal.rfft({0})",
            "Fftn": "tf.signal.fftNd({0})",
            "Ifft": "tf.signal.ifft({0})",
            "Ifftn": "tf.signal.ifftNd({0})",
            "Rfftn": "tf.signal.rfftNd({0})",
            "Irfftn": "tf.signal.irfftNd({0})",
            "Ifft2": "tf.signal.ifft2d({0})",
            "Rfft2": "tf.signal.rfft2d({0})",
            "Irfft2": "tf.signal.irfft2d({0})",
            "Hfft": "tf.signal.hfft({0})",
            "Rfftfreq": "tf.signal.rfftfreq({0}, d={d})",
        }


class TensorFlowControlFlowMixin:
    """Control Flow / Array / NN Mixin."""

    def _get_nn_ops(self, kwargs: dict) -> dict[str, str]:
        """Evaluate _get_nn_ops operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
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
            "OneHot": "tf.one_hot({0}, depth={depth})",
            "Clip": "tf.clip_by_value({0}, clip_value_min={a_min}, clip_value_max={a_max})",
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

    def _get_array_ops(self, kwargs: dict) -> dict[str, str]:
        """Evaluate _get_array_ops operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
        return {
            "BroadcastInDim": "{0}.broadcast_in_dim({1}, {2})",
            "ConvGeneralDilated": "{0}.conv_general_dilated({1}, {2})",
            "DotGeneral": "{0}.dot_general({1}, {2})",
            "DynamicSlice": "{0}.dynamic_slice({1}, {2})",
            "DynamicUpdateSlice": "{0}.dynamic_update_slice({1}, {2})",
            "Pmean": "{0}.pmean({1})",
            "Psum": "{0}.psum({1})",
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
