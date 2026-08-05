# ruff: noqa: E501
"""Mixins for MLX."""

from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)


class MLXVisionAudioMixin:
    """Vision and Audio Mixin."""


class MLXShapeMixin:
    """Shape and Formatting Mixin."""


class MLXOpRegistryMixin:
    """Op Registry Mixin."""

    _SIMPLE_OPS_MAP = {
        "BroadcastInDim": "{0}.broadcast_in_dim({1}, {2})",
        "ConvGeneralDilated": "{0}.conv_general_dilated({1}, {2})",
        "DotGeneral": "{0}.dot_general({1}, {2})",
        "DynamicSlice": "{0}.dynamic_slice({1}, {2})",
        "DynamicUpdateSlice": "{0}.dynamic_update_slice({1}, {2})",
        "Pmean": "{0}.pmean({1})",
        "Psum": "{0}.psum({1})",
        "Matmul": "mx.matmul({0}, {1})",
        "Trace": "mx.trace({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "Outer": "mx.outer({0}, {1})",
        "Svdvals": "mx.linalg.svd({0}, compute_uv=False)",
        "Tensordot": "mx.tensordot({0}, {1}, axes={axes})",
        "Tensorinv": "mx.linalg.tensorinv({0}, ind={ind})",
        "Tensorsolve": "mx.linalg.tensorsolve({0}, {1}, axes={axes})",
        "Vecdot": "mx.sum({0} * {1}, axis={axis})",
        "Adjoint": "tf.linalg.adjoint",
        "LuMatrixInverse": "mx.linalg.inv(mx.take_along_axis(mx.matmul(mx.tril({0}, -1) + mx.eye({0}.shape[-1], dtype={0}.dtype), mx.triu({0})), mx.broadcast_to(mx.expand_dims(mx.argsort({1}, axis=-1), -1), {0}.shape), axis=-2))",
        "LuReconstruct": "mx.take_along_axis(mx.matmul(mx.tril({0}, -1) + mx.eye({0}.shape[-1], dtype={0}.dtype), mx.triu({0})), mx.broadcast_to(mx.expand_dims(mx.argsort({1}, axis=-1), -1), {0}.shape), axis=-2)",
        "BandPart": "tf.linalg.band_part",
        "CholeskySolve": "mx.linalg.solve(mx.matmul({0}, mx.swapaxes({0}, -1, -2)), {1})",
        "Dot": "mx.dot({0}, {1})",
        "BroadcastTo": "mx.broadcast_to({0}, {shape})",
        "Reshape": "mx.reshape({0}, {shape})",
        "TruncateDiv": "mx.trunc(mx.divide({0}, {1}))",
        "TruncateMod": "mx.remainder({0}, {1})",
        "TrueDivide": "mx.divide({0}, {1})",
        "Sigmoid": "mx.sigmoid({0})",
        "Softmax": "mx.softmax({0}, axis={axis})",
        "LogSoftmax": "mx.log_softmax({0}, axis={axis})",
        "OneHot": "mx.eye({depth})[{0}]",
        "Clip": "mx.clip({0}, a_min={a_min}, a_max={a_max})",
        "Arange": "mx.arange({0})",
        "Frombuffer": "mx.frombuffer({0}, dtype={dtype}, count={count}, offset={offset})",
        "Sort": "mx.sort({0}, axis={dimension})",
        "ArgSort": "mx.argsort({0}, axis={dimension})",
        "Allclose": "mx.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
        "Fftnd": "mx.fft.fftn({0})",
        "Ifftnd": "mx.fft.ifftn({0})",
        "Rfftnd": "mx.fft.rfftn({0})",
        "Irfftnd": "mx.fft.irfftn({0})",
        "Fftshift": "mx.fft.fftshift({0})",
        "Ifftshift": "mx.fft.ifftshift({0})",
        "Fft": "mx.fft.fft({0})",
        "Rfft": "mx.fft.rfft({0})",
        "Fftn": "mx.fft.fftn({0})",
        "Ifft": "mx.fft.ifft({0})",
        "Ifftn": "mx.fft.ifftn({0})",
        "Rfftn": "mx.fft.rfftn({0})",
        "Irfftn": "mx.fft.irfftn({0})",
        "Ifft2": "mx.fft.ifft2({0})",
        "Rfft2": "mx.fft.rfft2({0})",
        "Irfft2": "mx.fft.irfft2({0})",
        "Hfft": "mx.fft.hfft({0})",
        "Rfftfreq": "mx.fft.rfftfreq({0}, d={d})",
        "Erfinv": "mx.erfinv({0})",
        "NanToNum": "mx.where(mx.isnan({0}), {nan}, mx.where(mx.isposinf({0}), {posinf}, mx.where(mx.isneginf({0}), {neginf}, {0})))",
        "AssignVariable": "{0}",
        "StopGradient": "mx.stop_gradient({0})",
        "Resize": "mx.image.resize({0}, {size})",
        "AffineGrid": "mx.image.affine_grid({0}, {size})",
        "GridSample": "mx.image.grid_sample({0}, {1})",
        "DrawBoundingBoxes": "{0}",
        "RgbToYiq": "mx.image.rgb_to_yiq({0})",
        "YiqToRgb": "mx.image.yiq_to_rgb({0})",
        "RgbToYuv": "mx.image.rgb_to_yuv({0})",
        "YuvToRgb": "mx.image.yuv_to_rgb({0})",
        "Fft2d": "mx.fft.fft2({0}, s={s}, axes={axes})",
        "Ifft2d": "mx.fft.ifft2({0}, s={s}, axes={axes})",
        "Fft3d": "mx.fft.fftn({0}, s={s}, axes={axes})",
        "Ifft3d": "mx.fft.ifftn({0}, s={s}, axes={axes})",
        "Rfft2d": "mx.fft.rfft2({0}, s={s}, axes={axes})",
        "Rfft3d": "mx.fft.rfftn({0}, s={s}, axes={axes})",
        "Irfft": "mx.fft.irfft({0}, n={n}, axis={axis})",
        "Irfft2d": "mx.fft.irfft2({0}, s={s}, axes={axes})",
        "Irfft3d": "mx.fft.irfftn({0}, s={s}, axes={axes})",
        "HannWindow": "mx.linalg.hann_window({window_length})",
        "ReadVariable": "{0}",
        "Cholesky": "mx.linalg.cholesky({0})",
        "Svd": "mx.linalg.svd({0})",
        "Qr": "mx.linalg.qr({0})",
        "Inv": "mx.linalg.inv({0})",
        "Solve": "mx.linalg.solve({0}, {1})",
        "Eigvalsh": "mx.linalg.eigvalsh({0})",
        "Cond": "mx.linalg.cond({0}, p={p})",
        "Lstsq": "mx.linalg.lstsq({0}, {1}, rcond={rcond})[0]",
        "MatrixNorm": "mx.linalg.norm({0}, keepdims={keepdims})",
        "VectorNorm": "mx.linalg.norm({0}, axis={axis}, keepdims={keepdims}, ord={ord})",
        "MatrixRank": "mx.linalg.matrix_rank({0}, tol={tol}, hermitian={hermitian})",
        "MatrixTranspose": "mx.swapaxes({0}, -1, -2)",
        "MultiDot": "mx.linalg.multi_dot({0})",
        "Diagonal": "mx.diagonal({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "Eigh": "mx.linalg.eigh({0})",
        "Eig": "mx.linalg.eig({0})",
        "MatrixPower": "mx.linalg.matrix_power({0}, {n})",
        "Norm": "mx.linalg.norm({0}, ord={ord}, axis={axis}, keepdims={keepdims})",
        "Det": "mx.linalg.det({0})",
        "Slogdet": "mx.linalg.slogdet({0})",
        "Poly": "mx.poly({0})",
        "Polyadd": "mx.polyadd({0})",
        "Polyder": "mx.polyder({0})",
        "Polydiv": "mx.polydiv({0})",
        "Polyfit": "mx.polyfit({0})",
        "Polyint": "mx.polyint({0})",
        "Polymul": "mx.polymul({0})",
        "Polysub": "mx.polysub({0})",
        "Polyval": "mx.polyval({0})",
        "Roots": "mx.roots({0})",
        "BroadcastedIota": "mx.broadcasted_iota({0})",
        "Bincount": "mx.bincount({0})",
        "Histogram": "mx.histogram({0})",
        "Histogram2d": "mx.histogram2d({0})",
        "HistogramBinEdges": "mx.histogram_bin_edges({0})",
        "Histogramdd": "mx.histogramdd({0})",
        "Geomspace": "mx.geomspace({0})",
        "Gradient": "mx.gradient({0})",
        "I0": "mx.i0({0})",
        "Mgrid": "mx.mgrid({0})",
        "Ogrid": "mx.ogrid({0})",
        "R_": "mx.r_({0})",
        "C_": "mx.c_({0})",
        "Fromfile": "mx.fromfile({0})",
        "Fromfunction": "mx.fromfunction({0})",
        "Fromiter": "mx.fromiter({0})",
        "Frompyfunc": "mx.frompyfunc({0})",
        "Fromstring": "mx.fromstring({0})",
        "Pinv": "mx.linalg.pinv({0})",
        "TriInv": "mx.linalg.inv({0})",
        "TridiagonalSolve": "mx.linalg.solve(mx.diag({1}) + mx.diag({0}[..., 1:], k=-1) + mx.diag({2}[..., :-1], k=1), {3})",
        "TridiagonalMatmul": "mx.expand_dims({1}, -1) * {3} + mx.expand_dims(mx.concatenate([mx.zeros_like({0}[..., :1]), {0}[..., 1:]], axis=-1), -1) * mx.concatenate([mx.zeros_like({3}[..., :1, :]), {3}[..., :-1, :]], axis=-2) + mx.expand_dims(mx.concatenate([{2}[..., :-1], mx.zeros_like({2}[..., -1:])], axis=-1), -1) * mx.concatenate([{3}[..., 1:, :], mx.zeros_like({3}[..., -1:, :])], axis=-2)",  # noqa: E501
        "TriangularSolve": "mx.linalg.solve_triangular({0}.swapaxes(-1, -2).conjugate() if {adjoint} else {0}, {1}, upper=not {lower} if not {adjoint} else {lower})",
        "Lu": "mx.linalg.lu({0})",
        "LuFactor": "mx.linalg.lu_factor({0})",
        "LuSolve": "mx.linalg.lu_solve({0}, {1})",
        "MatrixExponential": "mx.linalg.matrix_exp({0})",
        "Cross": "mx.linalg.cross({0}, {1}, axis={axis})",
        "GatherNd": "mx.gather_nd({0}, {1})",
        "ScatterNd": "mx.scatter_nd({0}, {1}, {shape})",
        "TensorScatterUpdate": "(lambda c, i, u: [c.__setitem__(tuple(i[..., d] for d in range(i.shape[-1])), u), c][1])(mx.array({0}), {1}, {2})",
        "TensorScatterAdd": "(lambda c, i, u: [c.__setitem__(tuple(i[..., d] for d in range(i.shape[-1])), c[tuple(i[..., d] for d in range(i.shape[-1]))] + u), c][1])(mx.array({0}), {1}, {2})",
        "TensorScatterMax": "(lambda c, i, u: [c.__setitem__(tuple(i[..., d] for d in range(i.shape[-1])), mx.maximum(c[tuple(i[..., d] for d in range(i.shape[-1]))], u)), c][1])(mx.array({0}), {1}, {2})",
        "TensorScatterMin": "(lambda c, i, u: [c.__setitem__(tuple(i[..., d] for d in range(i.shape[-1])), mx.minimum(c[tuple(i[..., d] for d in range(i.shape[-1]))], u)), c][1])(mx.array({0}), {1}, {2})",
        "Scatter": "mx.scatter({0}, {1}, {2}, {dim})",
        "ScatterAdd": "mx.scatter_add({0}, {1}, {2}, {dim})",
        "Gather": "mx.take({0}, {1}, {dim})",
        "PRNGKey": "mx.random.key({seed})",
        "RandomSplit": "mx.random.split({0}, {num})",
        "RandomUniform": "mx.random.uniform(low={minval}, high={maxval}, shape={shape}, key={0})",
        "RandomNormal": "mx.random.normal(shape={shape}, key={0})",
        "RandomRandint": "mx.random.randint(low={minval}, high={maxval}, shape={shape}, key={0})",
        "RandomBernoulli": "mx.random.bernoulli(p={p}, shape={shape}, key={0})",
    }

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Retrieve the dictionary mapping operation names to their corresponding MLX format strings.

        Args:
            kwargs: A dictionary of operation keyword arguments.

        Returns:
            A dictionary mapping operation type names to format strings.
        """
        ops_map = self._SIMPLE_OPS_MAP.copy()
        ops_map["Beta"] = "mx.random.uniform(shape={shape})"
        ops_map["Dirichlet"] = "mx.random.uniform(shape={shape})"
        ops_map["Gamma"] = "mx.random.uniform(shape={shape})"
        ops_map["RngBitGenerator"] = "mx.random.randint(0, 255, {shape})"
        ops_map["RngUniform"] = "mx.random.uniform(low={0}, high={1}, shape={shape})"
        ops_map["Infeed"] = "{0}"
        ops_map["Outfeed"] = "{0}"
        ops_map["AxisIndex"] = "0"
        ops_map["AllToAll"] = "{0}"
        ops_map["Pmax"] = "{0}"
        ops_map["Pmin"] = "{0}"
        ops_map["PsumScatter"] = "{0}"
        ops_map["Pswapaxes"] = "{0}"
        ops_map["Ppermute"] = "{0}"
        ops_map["Pshuffle"] = "{0}"
        ops_map["CreateToken"] = "0"
        ops_map["WithShardingConstraint"] = "{0}"
        return ops_map


class MLXNNOpsVisitor:
    """MLX NN ops visitor mixin."""

    def visit_Rope(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Rope operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Rope operation.
        """
        dim = node.attributes.get("dim")
        traditional = node.attributes.get("traditional", False)
        base = node.attributes.get("base", 10000.0)
        scale = node.attributes.get("scale", 1.0)
        offset = node.attributes.get("offset", 0)
        return f"mx.fast.rope({input_vars[0]}, {dim}, traditional={traditional}, base={base}, scale={scale}, offset={offset})"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the PowerIteration operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the PowerIteration operation.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"mlx_power_iteration({input_vars[0]}, {num_iters}, {u_var})"


class MLXVisionVisitor:
    """MLX Vision ops visitor mixin."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the ElasticTransform operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ElasticTransform operation.
        """
        (interpolation, fill_value, data_format) = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the GaussianBlur operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the GaussianBlur operation.
        """
        (kernel_size, sigma, padding, data_format) = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the MedianFilter operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the MedianFilter operation.
        """
        (kernel_size, sigma, padding, data_format) = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the ExtractBoundingBoxes operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ExtractBoundingBoxes operation.
        """
        (crop_size, interpolation, extrapolation_value, data_format) = _extract_extract_boxes_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the IoU operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the IoU operation.
        """
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"mlx_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the NonMaxSuppression operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the NonMaxSuppression operation.
        """
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"mlx_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the ResizeBicubic operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ResizeBicubic operation.
        """
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"mlx_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the ResizeLanczos3 operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ResizeLanczos3 operation.
        """
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"mlx_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the PerspectiveTransform operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the PerspectiveTransform operation.
        """
        (interpolation, fill_value, data_format) = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"


class MLXAudioVisitor:
    """MLX Audio ops visitor mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Istft operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Istft operation.
        """
        from ml_switcheroo_compiler.backends.common.audio_utils import extract_stft_attributes

        (frame_length, frame_step, _, window, center, fft_len_str) = extract_stft_attributes(node)
        return f"mlx_istft({input_vars[0]}, STFTConfig({frame_length}, {frame_step}, {fft_len_str}, '{window}', {center}))"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the MelFilterbank operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the MelFilterbank operation.
        """
        num_mel_bins = node.attributes.get("num_mel_bins")
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
        sample_rate = node.attributes.get("sample_rate")
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")
        return f"mlx_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Mfcc operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Mfcc operation.
        """
        sample_rate = node.attributes.get("sample_rate")
        num_mel_bins = node.attributes.get("num_mel_bins", 40)
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
        num_mfccs = node.attributes.get("num_mfccs", 13)
        return f"mlx_mfcc({input_vars[0]}, MFCCConfig({sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs}))"


class MLXShapeOpsVisitor:
    """Visitor for MLX shape operations."""

    def _format_einsum(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        """Format the MLX code for the einsum operation.

        Args:
            input_vars (list): The input_vars parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            str: Result.
        """
        equation = kwargs.get("equation", "")
        if "operands" in kwargs:
            operands = kwargs["operands"]
            if isinstance(operands, list):
                # Unpack operands
                return f"mx.einsum('{equation}', *{input_vars[0]})"
        return f"mx.einsum('{equation}', {', '.join(input_vars)})"

    def _format_zeros_ones(self, op: str, kwargs: dict[str, object]) -> str:
        """Format the MLX code for the zeros_ones operation.

        Args:
            op: The string name of the operation (e.g., "zeros" or "ones").
            kwargs: A dictionary of keyword arguments.

        Returns:
            A string representing the formatted MLX code for zeros_ones.
        """
        shape = kwargs.get("shape", ())
        if isinstance(shape, list):
            shape = tuple(shape)
        dtype = kwargs.get("dtype", "float32")
        dt = f"mx.{dtype}" if dtype else "None"
        return f"mx.{op}({shape}, dtype={dt})"

    def _format_full(self, kwargs: dict[str, object]) -> str:
        """Format the MLX code for the full operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            str: Result.
        """
        shape = kwargs.get("shape", ())
        if isinstance(shape, list):
            shape = tuple(shape)
        fill_value = kwargs.get("fill_value", 0.0)
        dtype = kwargs.get("dtype", "float32")
        dt = f"mx.{dtype}" if dtype else "None"
        return f"mx.full({shape}, {fill_value}, dtype={dt})"

    def _format_transpose(self, kwargs: dict[str, object]) -> str:
        """Format the MLX code for the transpose operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            str: Result.
        """
        axes = kwargs.get("axes", None)
        return f"axes={axes}" if axes is not None else ""

    def _format_random_categorical(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        """Format the MLX code for the random_categorical operation.

        Args:
            input_vars (list): The input_vars parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            str: Result.
        """
        num_samples = kwargs.get("num_samples", 1)
        return f"mx.random.categorical({input_vars[0]}, num_samples={num_samples})"

    def visit_Concatenate(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Concatenate operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Concatenate operation.
        """
        axis = node.attributes.get("axis", 0)
        return f"mx.concatenate({input_vars[0]}, axis={axis})"

    def visit_Stack(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Stack operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Stack operation.
        """
        axis = node.attributes.get("axis", 0)
        return f"mx.stack({input_vars[0]}, axis={axis})"

    def visit_Partition(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Partition operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Partition operation.
        """
        axis = node.attributes.get("axis", -1)
        kth = node.attributes.get("kth", 0)
        return f"mx.partition({input_vars[0]}, {kth}, axis={axis})"

    def visit_Argpartition(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Argpartition operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Argpartition operation.
        """
        axis = node.attributes.get("axis", -1)
        kth = node.attributes.get("kth", 0)
        return f"mx.argpartition({input_vars[0]}, {kth}, axis={axis})"

    def visit_Repeat(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Repeat operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Repeat operation.
        """
        repeats = node.attributes.get("repeats", 1)
        axis = node.attributes.get("axis", None)
        return f"mx.repeat({input_vars[0]}, {repeats}, axis={axis})"

    def visit_Roll(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Roll operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Roll operation.
        """
        shift = node.attributes.get("shift", 1)
        axis = node.attributes.get("axis", None)
        return f"mx.roll({input_vars[0]}, {shift}, axis={axis})"

    def visit_Tile(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Tile operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Tile operation.
        """
        reps = node.attributes.get("reps", 1)
        return f"mx.tile({input_vars[0]}, {reps})"

    def visit_TopK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the TopK operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the TopK operation.
        """
        k = node.attributes.get("k", 1)
        axis = node.attributes.get("axis", -1)
        return f"mx.topk({input_vars[0]}, {k}, axis={axis})"

    def visit_Moveaxis(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Moveaxis operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Moveaxis operation.
        """
        source = node.attributes.get("source", 0)
        destination = node.attributes.get("destination", 1)
        return f"mx.moveaxis({input_vars[0]}, {source}, {destination})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the RaggedDot operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the RaggedDot operation.
        """
        return f"mx.matmul({input_vars[0]}, {input_vars[1]})"

    def visit_NanToNum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the NanToNum operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the NanToNum operation.
        """
        nan = node.attributes.get("nan", 0.0)
        posinf = node.attributes.get("posinf", None)
        neginf = node.attributes.get("neginf", None)
        return f"mx.nan_to_num({input_vars[0]}, nan={nan}, posinf={posinf}, neginf={neginf})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Einsum operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Einsum operation.
        """
        return self._format_einsum(input_vars, node.attributes)

    def visit_Zeros(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Zeros operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Zeros operation.
        """
        return self._format_zeros_ones("zeros", node.attributes)

    def visit_Ones(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Ones operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Ones operation.
        """
        return self._format_zeros_ones("ones", node.attributes)

    def visit_Full(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Full operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Full operation.
        """
        return self._format_full(node.attributes)

    def visit_ConstantOfShape(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the ConstantOfShape operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ConstantOfShape operation.
        """
        return self._format_full(node.attributes)

    def visit_Transpose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the Transpose operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Transpose operation.
        """
        args_str = self._format_transpose(node.attributes)
        return f"mx.transpose({input_vars[0]}, {args_str})" if args_str else f"mx.transpose({input_vars[0]})"

    def visit_RandomCategorical(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate MLX code for the RandomCategorical operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the RandomCategorical operation.
        """
        return self._format_random_categorical(input_vars, node.attributes)
