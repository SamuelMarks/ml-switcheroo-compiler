"""Keras Target Emission."""

from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("keras")
class KerasCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """Emit Keras Functional API script from IR."""

    def _get_backend_prefix(self) -> str:
        return "keras"

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes the object.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        res = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        res = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs: object) -> str:
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"
        return "keras.ops.transpose({0})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size = node.attributes.get("kernel_size")
        sigma = node.attributes.get("sigma")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return (
            f"keras_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"
        )

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size = node.attributes.get("kernel_size")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size = node.attributes.get("crop_size")
        interpolation = node.attributes.get("interpolation", "bilinear")
        extrapolation_value = node.attributes.get("extrapolation_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"keras_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"keras_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"keras_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"keras_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length = node.attributes.get("frame_length")
        frame_step = node.attributes.get("frame_step")
        fft_length = node.attributes.get("fft_length", None)
        window = node.attributes.get("window", "hann")
        center = node.attributes.get("center", True)
        fft_len_str = "None" if fft_length is None else str(fft_length)
        return f"keras_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins = node.attributes.get("num_mel_bins")
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
        sample_rate = node.attributes.get("sample_rate")
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")
        return f"keras_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        sample_rate = node.attributes.get("sample_rate")
        num_mel_bins = node.attributes.get("num_mel_bins", 40)
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
        num_mfccs = node.attributes.get("num_mfccs", 13)
        return f"keras_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        """Generate keras.ops.image.perspective_transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras.ops.image.perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, interpolation='{interpolation}', fill_value={fill_value}, data_format={df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"keras.ops.einsum('{eq}', {args_str})"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

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
            "Arange": "keras.ops.arange({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "keras.ops.sort({0}, axis={dimension})",
            "ArgSort": "keras.ops.argsort({0}, axis={dimension})",
            "Allclose": "keras.ops.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "keras.ops.fft.fft({0})",
            "Rfft": "keras.ops.fft.rfft({0})",
            "Fftn": "keras.ops.fft.fftn({0})",
            "Erfinv": "keras.ops.erfinv({0})",
            "NanToNum": "keras.ops.where(keras.ops.isnan({0}), {nan}, keras.ops.where(keras.ops.isinf({0}) & ({0} > 0), {posinf}, keras.ops.where(keras.ops.isinf({0}) & ({0} < 0), {neginf}, {0})))",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": self._format_transpose(kwargs),
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(
                prefix="keras.ops", op_type=op_type, input_vars=input_vars, kwargs=kwargs
            )
        )

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

    def _emit_body_return(self, returns: list[str]) -> None:
        pass

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
            "def keras_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):",
            "    import tensorflow as tf",
            "    return keras.ops.convert_to_tensor(tf.signal.linear_to_mel_weight_matrix(",
            "        num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=sample_rate,",
            "        lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz))",
            "def keras_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):",
            "    import tensorflow as tf",
            "    spec_tf = keras.ops.convert_to_tensor(spectrogram)",
            "    mel_weights = tf.signal.linear_to_mel_weight_matrix(",
            "        num_mel_bins=num_mel_bins, num_spectrogram_bins=spec_tf.shape[-1], sample_rate=sample_rate,",
            "        lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz)",
            "    mel_spec = tf.matmul(spec_tf, mel_weights)",
            "    log_mel = tf.math.log(mel_spec + 1e-6)",
            "    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :num_mfccs]",
            "    return keras.ops.convert_to_tensor(mfccs)",
            "def keras_istft(stft_tensor, frame_length, frame_step, fft_length, window, center):",
            "    import tensorflow as tf",
            "    stft_tensor = keras.ops.convert_to_tensor(stft_tensor)",
            "    if fft_length is None: fft_length = frame_length",
            "    if window == 'hann': win = tf.signal.hann_window(frame_length, periodic=True)",
            "    elif window == 'hamming': win = tf.signal.hamming_window(frame_length, periodic=True)",
            "    else: win = None",
            "    res = tf.signal.inverse_stft(stft_tensor, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, window_fn=lambda w, d: win)",
            "    return keras.ops.convert_to_tensor(res)",
            "def keras_resize(images, size, interpolation, align_corners):",
            "    import tensorflow as tf",
            "    method = tf.image.ResizeMethod.LANCZOS3 if interpolation == 'lanczos3' else tf.image.ResizeMethod.BICUBIC",
            "    images_tf = keras.ops.convert_to_tensor(images)",
            "    out = tf.image.resize(images_tf, size, method=method, antialias=True)",
            "    return keras.ops.convert_to_tensor(out)",
            "def keras_iou(boxes1, boxes2, bounding_box_format):",
            "    from ml_switcheroo_compiler.backends.eager_utils import iou_eager",
            "    import keras.ops as kops",
            "    return iou_eager(kops, boxes1, boxes2, bounding_box_format)",
            "def keras_nms(boxes, scores, max_output_size, iou_threshold, score_threshold):",
            "    import tensorflow as tf",
            "    boxes_tf = keras.ops.convert_to_tensor(boxes)",
            "    scores_tf = keras.ops.convert_to_tensor(scores)",
            "    out = tf.image.non_max_suppression(boxes_tf, scores_tf, max_output_size, iou_threshold, score_threshold)",
            "    return keras.ops.convert_to_tensor(out)",
            "def keras_extract_bounding_boxes(images, boxes, box_indices, crop_size, interpolation='bilinear', extrapolation_value=0.0, data_format=None):",
            "    import tensorflow as tf",
            "    images_tf = keras.ops.convert_to_tensor(images)",
            '    if data_format == "channels_first":',
            "        images_tf = keras.ops.transpose(images_tf, (0, 2, 3, 1))",
            "    boxes_tf = keras.ops.convert_to_tensor(boxes)",
            "    box_indices_tf = keras.ops.convert_to_tensor(box_indices)",
            "    out = tf.image.crop_and_resize(images_tf, boxes_tf, box_indices_tf, crop_size, method=interpolation, extrapolation_value=extrapolation_value)",
            "    out = keras.ops.convert_to_tensor(out)",
            '    if data_format == "channels_first":',
            "        out = keras.ops.transpose(out, (0, 3, 1, 2))",
            "    return out",
            "def keras_median_filter(images, kernel_size, padding='same', data_format=None):",
            "    orig_ndim = len(images.shape)",
            "    if orig_ndim == 3:",
            "        images = keras.ops.expand_dims(images, 0)",
            '    if data_format == "channels_first":',
            "        images = keras.ops.transpose(images, (0, 2, 3, 1))",
            "    import tensorflow as tf",
            "    ky, kx = kernel_size",
            "    # For median filter in Keras, we have to either extract patches or use tf.raw_ops. There's no built-in.",
            "    # We'll use tf.image.extract_patches.",
            "    images_tf = keras.ops.convert_to_tensor(images)",
            "    B, H, W, C = images_tf.shape",
            "    if padding == 'same':",
            "        pad_y, pad_x = ky // 2, kx // 2",
            "        images_tf = tf.pad(images_tf, [[0, 0], [pad_y, pad_y], [pad_x, pad_x], [0, 0]])",
            "        H, W = images_tf.shape[1], images_tf.shape[2]",
            "    out_H, out_W = H - ky + 1, W - kx + 1",
            "    patches = tf.image.extract_patches(images_tf, sizes=[1, ky, kx, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')",
            "    patches = tf.reshape(patches, [-1, out_H, out_W, ky * kx, C])",
            "    # tf.math.top_k to get median",
            "    sorted_patches = tf.sort(patches, axis=3)",
            "    out = sorted_patches[..., (ky * kx) // 2, :]",
            "    out = keras.ops.convert_to_tensor(out)",
            '    if data_format == "channels_first":',
            "        out = keras.ops.transpose(out, (0, 3, 1, 2))",
            "    if orig_ndim == 3:",
            "        out = out[0]",
            "    return out",
            "def keras_gaussian_blur(images, kernel_size, sigma, padding='same', data_format=None):",
            "    orig_ndim = len(images.shape)",
            "    if orig_ndim == 3:",
            "        images = keras.ops.expand_dims(images, 0)",
            '    if data_format == "channels_first":',
            "        images = keras.ops.transpose(images, (0, 2, 3, 1))",
            "    B, H, W, C = images.shape",
            "    ky, kx = kernel_size",
            "    sy, sx = sigma",
            "    y = keras.ops.arange(-ky // 2 + 1, ky // 2 + 1, dtype=images.dtype)",
            "    x = keras.ops.arange(-kx // 2 + 1, kx // 2 + 1, dtype=images.dtype)",
            "    yy, xx = keras.ops.meshgrid(y, x, indexing='ij')",
            "    kernel = keras.ops.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))",
            "    kernel = kernel / keras.ops.sum(kernel)",
            "    kernel = keras.ops.reshape(kernel, (ky, kx, 1, 1))",
            "    kernel = keras.ops.broadcast_to(kernel, (ky, kx, C, 1))",
            "    out = keras.ops.depthwise_conv(images, kernel, padding=padding)",
            '    if data_format == "channels_first":',
            "        out = keras.ops.transpose(out, (0, 3, 1, 2))",
            "    if orig_ndim == 3:",
            "        out = out[0]",
            "    return out",
            "def keras_elastic_transform(images, displacement, interpolation='bilinear', fill_value=0.0, data_format=None):",
            "    B_sz, H_dim, W_dim, C_dim = keras.ops.shape(images)",
            "    y_grid, x_grid = keras.ops.meshgrid(keras.ops.arange(H_dim), keras.ops.arange(W_dim), indexing='ij')",
            "    y_grid = keras.ops.cast(y_grid, images.dtype)",
            "    x_grid = keras.ops.cast(x_grid, images.dtype)",
            "    y_grid = keras.ops.broadcast_to(y_grid, (B_sz, H_dim, W_dim))",
            "    x_grid = keras.ops.broadcast_to(x_grid, (B_sz, H_dim, W_dim))",
            "    y = y_grid + displacement[..., 0]",
            "    x = x_grid + displacement[..., 1]",
            "    return keras.ops.image.map_coordinates(images, keras.ops.stack([y, x], axis=-1), order=1 if interpolation == 'bilinear' else 0, fill_mode='constant', fill_value=fill_value)",
        ]

        self.indent_level = 0
        self.add_line("def get_model():")
        self.indent_level += 1

        self.keras_input_vars = []
        self.keras_output_vars = []

        self._generate_body()

        inputs_str = ", ".join(self.keras_input_vars)
        outputs_str = ", ".join(self.keras_output_vars)
        self.add_line(
            f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])",
        )

        return "\n".join(self.code)
