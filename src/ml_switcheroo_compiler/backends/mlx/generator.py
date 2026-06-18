"""MLX Target Emission."""

from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("mlx")
class MLXCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """Emit MLX-compatible code from IR."""

    def _get_backend_prefix(self) -> str:
        return "mlx"

    _SIMPLE_OPS_MAP = {
        "Matmul": "mx.matmul({0}, {1})",
        "Dot": "mx.dot({0}, {1})",
        "BroadcastTo": "mx.broadcast_to({0}, {shape})",
        "Reshape": "mx.reshape({0}, {shape})",
        "TrueDivide": "mx.divide({0}, {1})",
        "Arange": "mx.arange({0})",
        "Sort": "mx.sort({0}, axis={dimension})",
        "ArgSort": "mx.argsort({0}, axis={dimension})",
        "Allclose": "mx.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
        "Fft": "mx.fft.fft({0})",
        "Rfft": "mx.fft.rfft({0})",
        "Fftn": "mx.fft.fftn({0})",
        "Erfinv": "mx.erfinv({0})",
        "NanToNum": "mx.where(mx.isnan({0}), {nan}, mx.where(mx.isposinf({0}), {posinf}, mx.where(mx.isneginf({0}), {neginf}, {0})))",
        "AssignVariable": "{0}",
        "ReadVariable": "{0}",
        "Cholesky": "mx.linalg.cholesky({0})",
        "Svd": "mx.linalg.svd({0})",
        "Qr": "mx.linalg.qr({0})",
        "Inv": "mx.linalg.inv({0})",
        "Solve": "mx.linalg.solve({0}, {1})",
        "Eigvalsh": "mx.linalg.eigvalsh({0})",
        "Eigh": "mx.linalg.eigh({0})",
        "MatrixPower": "mx.linalg.matrix_power({0}, {1})",
        "Norm": "mx.linalg.norm({0})",
        "Det": "mx.linalg.det({0})",
        "Slogdet": "mx.linalg.slogdet({0})",
        "Cross": "mx.linalg.cross({0}, {1})",
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

    def _format_einsum(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"mx.einsum('{eq}', {args_str})"

    def _format_zeros_ones(self, op: str, kwargs: dict[str, object]) -> str:
        dtype_str = f", dtype='{kwargs['dtype']}'" if "dtype" in kwargs else ""
        return f"mx.{op.lower()}({{shape}}){dtype_str}"

    def _format_full(self, kwargs: dict[str, object]) -> str:
        dtype_str = f", dtype='{kwargs['dtype']}'" if "dtype" in kwargs else ""
        return f"mx.full({{shape}}, {{fill_value}}){dtype_str}"

    def _format_transpose(self, kwargs: dict[str, object]) -> str:
        return "mx.transpose({0}, {axes})" if "axes" in kwargs else "mx.transpose({0})"

    def _format_random_categorical(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        if len(input_vars) > 1:
            return "mx.random.categorical(logits={1}, axis={axis}, shape={shape}, key={0})"
        return "mx.random.categorical(axis={axis}, shape={shape}, key={0})"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"mlx_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size = node.attributes.get("kernel_size")
        sigma = node.attributes.get("sigma")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size = node.attributes.get("kernel_size")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size = node.attributes.get("crop_size")
        interpolation = node.attributes.get("interpolation", "bilinear")
        extrapolation_value = node.attributes.get("extrapolation_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"mlx_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"mlx_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        # mlx has no bicubic resize. We fallback to map to eager utility
        return f"mlx_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"mlx_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length = node.attributes.get("frame_length")
        frame_step = node.attributes.get("frame_step")
        fft_length = node.attributes.get("fft_length", None)
        window = node.attributes.get("window", "hann")
        center = node.attributes.get("center", True)
        fft_len_str = "None" if fft_length is None else str(fft_length)
        return f"mlx_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins = node.attributes.get("num_mel_bins")
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
        sample_rate = node.attributes.get("sample_rate")
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")
        return f"mlx_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        sample_rate = node.attributes.get("sample_rate")
        num_mel_bins = node.attributes.get("num_mel_bins", 40)
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
        num_mfccs = node.attributes.get("num_mfccs", 13)
        return f"mlx_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum."""
        return self._format_einsum(input_vars, kwargs)

    def visit_Zeros(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Zeros."""
        fmt = self._format_zeros_ones("Zeros", kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def visit_Ones(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Ones."""
        fmt = self._format_zeros_ones("Ones", kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def visit_Full(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Full."""
        fmt = self._format_full(kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def visit_Transpose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Transpose."""
        fmt = self._format_transpose(kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def visit_RandomCategorical(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle RandomCategorical."""
        fmt = self._format_random_categorical(input_vars, kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated MLX Python code
        """
        op_type = getattr(node, "op_type", "")

        fmt = self._SIMPLE_OPS_MAP.get(op_type)

        if fmt is not None:
            return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(prefix="mx", op_type=op_type, input_vars=input_vars, kwargs=kwargs)
        )

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = mx.array({val_repr})")

    def generate(self) -> str:
        """Generate MLX model code from the IR graph.

        Returns:
            str: The generated MLX Python code
        """
        self.code = [
            self.header.strip(),
            "import mlx.core as mx",
            "import mlx.nn as nn\n",
            "def mx_group_norm(x, groups, weight=None, bias=None, axis=-1, epsilon=1e-5):",
            "    import mlx.core as mx",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = mx.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    mean = mx.mean(reshaped_x, axis=reduction_axes, keepdims=True)",
            "    var = mx.var(reshaped_x, axis=reduction_axes, keepdims=True)",
            "    normalized = (reshaped_x - mean) / mx.sqrt(var + epsilon)",
            "    out = mx.reshape(normalized, shape)",
            "    if weight is not None:",
            "        w_shape = [1] * ndims",
            "        w_shape[axis] = C",
            "        weight = mx.reshape(weight, w_shape)",
            "        out = out * weight",
            "    if bias is not None:",
            "        b_shape = [1] * ndims",
            "        b_shape[axis] = C",
            "        bias = mx.reshape(bias, b_shape)",
            "        out = out + bias",
            "    return out",
            "def mx_group_mean(x, groups, axis=-1):",
            "    import mlx.core as mx",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = mx.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return mx.mean(reshaped_x, axis=reduction_axes, keepdims=True)",
            "def mx_group_variance(x, groups, axis=-1):",
            "    import mlx.core as mx",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = mx.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return mx.var(reshaped_x, axis=reduction_axes, keepdims=True)",
            "def mlx_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):",
            "    from ml_switcheroo_compiler.backends.eager_utils import mel_filterbank_eager",
            "    import mlx.core as mx",
            "    return mel_filterbank_eager(mx, None, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)",
            "def mlx_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):",
            "    from ml_switcheroo_compiler.backends.eager_utils import mfcc_eager",
            "    import mlx.core as mx",
            "    return mfcc_eager(mx, spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs)",
            "def mlx_istft(stft_tensor, frame_length, frame_step, fft_length, window, center):",
            "    from ml_switcheroo_compiler.backends.eager_utils import istft_eager",
            "    import mlx.core as mx",
            "    return istft_eager(mx, stft_tensor, frame_length, frame_step, fft_length, window, center)",
            "def mlx_resize(images, size, interpolation, align_corners):",
            "    from ml_switcheroo_compiler.backends.eager_utils import resize_eager",
            "    import mlx.core as mx",
            "    return resize_eager(mx, images, size, interpolation, align_corners)",
            "def mlx_iou(boxes1, boxes2, bounding_box_format):",
            "    from ml_switcheroo_compiler.backends.eager_utils import iou_eager",
            "    import mlx.core as mx",
            "    return iou_eager(mx, boxes1, boxes2, bounding_box_format)",
            "def mlx_nms(boxes, scores, max_output_size, iou_threshold, score_threshold):",
            "    from ml_switcheroo_compiler.backends.eager_utils import nms_eager",
            "    import mlx.core as mx",
            "    return nms_eager(mx, boxes, scores, max_output_size, iou_threshold, score_threshold)",
            "def mlx_extract_bounding_boxes(images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format):",
            "    # We'll map back to eager utility here since complex interpolation requires meshgrids",
            "    from ml_switcheroo_compiler.backends.eager_utils import extract_bounding_boxes_eager",
            "    import mlx.core as mx",
            "    return extract_bounding_boxes_eager(mx, images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format)",
            "def mlx_median_filter(images, kernel_size, padding, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = mx.expand_dims(images, 0)",
            '    if data_format == "channels_first":',
            "        images = mx.transpose(images, (0, 2, 3, 1))",
            "    B, H, W, C = images.shape",
            "    ky, kx = kernel_size",
            "    if padding == 'same':",
            "        pad_y, pad_x = ky // 2, kx // 2",
            "        images = mx.pad(images, ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)))",
            "        H, W = images.shape[1], images.shape[2]",
            "    out_H, out_W = H - ky + 1, W - kx + 1",
            "    # Extract patches",
            "    strides = images.strides",
            "    patches = mx.as_strided(images, (B, out_H, out_W, ky, kx, C), (strides[0], strides[1], strides[2], strides[1], strides[2], strides[3]))",
            "    patches = mx.reshape(patches, (B, out_H, out_W, ky * kx, C))",
            "    # Sort and take median",
            "    sorted_patches = mx.sort(patches, axis=3)",
            "    out = sorted_patches[..., (ky * kx) // 2, :]",
            '    if data_format == "channels_first":',
            "        out = mx.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def mlx_gaussian_blur(images, kernel_size, sigma, padding, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = mx.expand_dims(images, 0)",
            '    if data_format == "channels_first":',
            "        images = mx.transpose(images, (0, 2, 3, 1))",
            "    B, H, W, C = images.shape",
            "    ky, kx = kernel_size",
            "    sy, sx = sigma",
            "    y = mx.arange(-ky // 2 + 1, ky // 2 + 1, dtype=images.dtype)",
            "    x = mx.arange(-kx // 2 + 1, kx // 2 + 1, dtype=images.dtype)",
            "    yy, xx = mx.meshgrid(y, x, indexing='ij')",
            "    kernel = mx.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))",
            "    kernel = kernel / mx.sum(kernel)",
            "    kernel = mx.reshape(kernel, (ky, kx, 1, 1))",
            "    kernel = mx.broadcast_to(kernel, (ky, kx, C, 1))",
            "    if padding == 'same':",
            "        pad_y, pad_x = ky // 2, kx // 2",
            "        images = mx.pad(images, ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)))",
            "    out = mx.conv2d(images, kernel, groups=C)",
            '    if data_format == "channels_first":',
            "        out = mx.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def mlx_elastic_transform(images, displacement, interpolation, fill_value, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = mx.expand_dims(images, 0)",
            "        displacement = mx.expand_dims(displacement, 0)",
            '    if data_format == "channels_first":',
            "        images = mx.transpose(images, (0, 2, 3, 1))",
            "    B, H, W, C = images.shape",
            "    y, x = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')",
            "    y, x = mx.broadcast_to(y, (B, H, W)), mx.broadcast_to(x, (B, H, W))",
            "    y, x = y + displacement[..., 0], x + displacement[..., 1]",
            "    y_valid = (y >= 0) & (y <= H - 1)",
            "    x_valid = (x >= 0) & (x <= W - 1)",
            "    valid = y_valid & x_valid",
            '    if interpolation == "nearest":',
            "        y_idx = mx.clip(mx.round(y), 0, H - 1).astype(mx.int32)",
            "        x_idx = mx.clip(mx.round(x), 0, W - 1).astype(mx.int32)",
            "        out = images[mx.arange(B)[:, None, None], y_idx, x_idx]",
            "        out = mx.where(mx.expand_dims(valid, -1), out, mx.array(fill_value, dtype=out.dtype))",
            "    else:",
            "        y0 = mx.clip(mx.floor(y), 0, H - 1).astype(mx.int32)",
            "        x0 = mx.clip(mx.floor(x), 0, W - 1).astype(mx.int32)",
            "        y1 = mx.clip(y0 + 1, 0, H - 1)",
            "        x1 = mx.clip(x0 + 1, 0, W - 1)",
            "        dy, dx = mx.expand_dims(y - y0, -1), mx.expand_dims(x - x0, -1)",
            "        b_idx = mx.arange(B)[:, None, None]",
            "        v00 = images[b_idx, y0, x0]",
            "        v01 = images[b_idx, y0, x1]",
            "        v10 = images[b_idx, y1, x0]",
            "        v11 = images[b_idx, y1, x1]",
            "        w00, w01, w10, w11 = (1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx",
            "        out = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11",
            "        out = mx.where(mx.expand_dims(valid, -1), out, mx.array(fill_value, dtype=out.dtype))",
            '    if data_format == "channels_first":',
            "        out = mx.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def mlx_power_iteration(w, num_iters, u=None):",
            "    import mlx.core as mx",
            "    if u is None:",
            "        u = mx.ones(w.shape[:-2] + [w.shape[-2], 1], dtype=w.dtype)",
            "    def body_fn(val):",
            "        i, u_curr, _ = val",
            "        w_t = mx.swapaxes(w, -1, -2)",
            "        v_next = mx.matmul(w_t, u_curr)",
            "        v_next = v_next / (mx.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)",
            "        u_next = mx.matmul(w, v_next)",
            "        u_next = u_next / (mx.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)",
            "        return i + 1, u_next, v_next",
            "    def cond_fn(val):",
            "        return val[0] < num_iters",
            "    init_v = mx.zeros(w.shape[:-2] + [w.shape[-1], 1], dtype=w.dtype)",
            "    _, u_final, v_final = mx.while_loop(cond_fn, body_fn)( (mx.array(0), u, init_v) )",
            "    sigma = mx.matmul(mx.swapaxes(u_final, -1, -2), mx.matmul(w, v_final))",
            "    return mx.squeeze(v_final, -1), mx.squeeze(u_final, -1), mx.squeeze(mx.squeeze(sigma, -1), -1)",
            "class CompiledModel(nn.Module):",
        ]
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")
        self.add_line("pass\n")
        self.indent_level -= 1

        self.add_line("def __call__(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
