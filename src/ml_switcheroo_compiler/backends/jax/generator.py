"""JAX/Flax Target Emission."""

from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("jax")
class JAXCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """JAX code generator."""

    def _get_backend_prefix(self) -> str:
        return "jax"

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        axis_name = node.attributes.get("axis_name", "'x'")
        return f"jax.lax.all_gather({tensor}, axis_name={axis_name})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        axis = node.attributes.get("axis", 0)
        axis_name = node.attributes.get("axis_name", "'x'")
        op = node.attributes.get("op", "jax.lax.psum")
        return f"jax.lax.reduce_scatter({tensor}, {op}, scatter_dimension={axis}, axis_name={axis_name})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        axis_name = node.attributes.get("axis_name", "'x'")
        op = node.attributes.get("op", "psum")
        return f"jax.lax.{op}({tensor}, axis_name={axis_name})"

    """Emit JAX-compatible pure functions from IR."""

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        res = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        res = "jnp.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"jax_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size = node.attributes.get("kernel_size")
        sigma = node.attributes.get("sigma")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size = node.attributes.get("kernel_size")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size = node.attributes.get("crop_size")
        interpolation = node.attributes.get("interpolation", "bilinear")
        extrapolation_value = node.attributes.get("extrapolation_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"jax_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"jax_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length = node.attributes.get("frame_length")
        frame_step = node.attributes.get("frame_step")
        fft_length = node.attributes.get("fft_length", None)
        window = node.attributes.get("window", "hann")
        center = node.attributes.get("center", True)
        fft_len_str = "None" if fft_length is None else str(fft_length)
        return f"jax_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins = node.attributes.get("num_mel_bins")
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
        sample_rate = node.attributes.get("sample_rate")
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")
        return f"jax_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        sample_rate = node.attributes.get("sample_rate")
        num_mel_bins = node.attributes.get("num_mel_bins", 40)
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
        num_mfccs = node.attributes.get("num_mfccs", 13)
        return f"jax_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"jnp.einsum('{eq}', {args_str})"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated JAX Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "jnp.matmul({0}, {1})",
            "Dot": "jnp.dot({0}, {1})",
            "BroadcastTo": "jnp.broadcast_to({0}, {shape})",
            "Reshape": "jnp.reshape({0}, {shape})",
            "TrueDivide": "jnp.true_divide({0}, {1})",
            "Arange": "jnp.arange({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "jnp.sort({0}, axis={dimension})",
            "ArgSort": "jnp.argsort({0}, axis={dimension})",
            "Allclose": "jnp.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "jnp.fft.fft({0})",
            "Rfft": "jnp.fft.rfft({0})",
            "Fftn": "jnp.fft.fftn({0})",
            "Erfinv": "jax.scipy.special.erfinv({0})",
            "NanToNum": "jnp.nan_to_num({0}, nan={nan}, posinf={posinf}, neginf={neginf})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "TensorScatterUpdate": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].set({2})",
            "TensorScatterAdd": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].add({2})",
            "TensorScatterMax": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].max({2})",
            "TensorScatterMin": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].min({2})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(prefix="jnp", op_type=op_type, input_vars=input_vars, kwargs=kwargs)
        )

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def generate(self) -> str:
        """Generate JAX code from the IR graph.

        Returns:
            str: The generated JAX Python code
        """
        self.code = [
            self.header.strip(),
            "import jax",
            "import jax.numpy as jnp",
            "def jax_group_norm(x, groups, weight=None, bias=None, axis=-1, epsilon=1e-5):",
            "    import jax.numpy as jnp",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = jnp.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    mean = jnp.mean(reshaped_x, axis=reduction_axes, keepdims=True)",
            "    var = jnp.var(reshaped_x, axis=reduction_axes, keepdims=True)",
            "    normalized = (reshaped_x - mean) / jnp.sqrt(var + epsilon)",
            "    out = jnp.reshape(normalized, shape)",
            "    if weight is not None:",
            "        w_shape = [1] * ndims",
            "        w_shape[axis] = C",
            "        weight = jnp.reshape(weight, w_shape)",
            "        out = out * weight",
            "    if bias is not None:",
            "        b_shape = [1] * ndims",
            "        b_shape[axis] = C",
            "        bias = jnp.reshape(bias, b_shape)",
            "        out = out + bias",
            "    return out",
            "def jax_group_mean(x, groups, axis=-1):",
            "    import jax.numpy as jnp",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = jnp.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return jnp.mean(reshaped_x, axis=reduction_axes, keepdims=True)",
            "def jax_group_variance(x, groups, axis=-1):",
            "    import jax.numpy as jnp",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = jnp.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return jnp.var(reshaped_x, axis=reduction_axes, keepdims=True)",
            "import jax.scipy.special\n",
            "",
            "def jax_elastic_transform(images, displacement, interpolation, fill_value, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = images[None, ...]",
            "        displacement = displacement[None, ...]",
            '    if data_format == "channels_first":',
            "        images = jnp.transpose(images, (0, 2, 3, 1))",
            "    B_sz, H_dim, W_dim, C_dim = images.shape",
            "    y_grid, x_grid = jnp.meshgrid(jnp.arange(H_dim), jnp.arange(W_dim), indexing='ij')",
            "    y_grid, x_grid = y_grid.astype(jnp.float32), x_grid.astype(jnp.float32)",
            "    import jax",
            "    import jax.scipy.ndimage as ndimage",
            "    def process_batch(img, disp):",
            "        src_y = y_grid + disp[..., 0]",
            "        src_x = x_grid + disp[..., 1]",
            "        def process_channel(c_img):",
            '            order = 1 if interpolation == "bilinear" else 0',
            "            return ndimage.map_coordinates(c_img, [src_y, src_x], order=order, mode='constant', cval=fill_value)",
            "        return jax.vmap(process_channel, in_axes=-1, out_axes=-1)(img)",
            "    out = jax.vmap(process_batch)(images, displacement)",
            '    if data_format == "channels_first":',
            "        out = jnp.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def jax_gaussian_blur(images, kernel_size, sigma, padding, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = images[None, ...]",
            '    if data_format == "channels_first":',
            "        images = jnp.transpose(images, (0, 2, 3, 1))",
            "    B, H, W, C = images.shape",
            "    ky, kx = kernel_size",
            "    sy, sx = sigma",
            "    y = jnp.arange(-ky // 2 + 1, ky // 2 + 1, dtype=images.dtype)",
            "    x = jnp.arange(-kx // 2 + 1, kx // 2 + 1, dtype=images.dtype)",
            "    yy, xx = jnp.meshgrid(y, x, indexing='ij')",
            "    kernel = jnp.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))",
            "    kernel = kernel / jnp.sum(kernel)",
            "    kernel = kernel.reshape(ky, kx, 1, 1)",
            "    kernel = jnp.broadcast_to(kernel, (ky, kx, C, 1))",
            "    import jax.lax as lax",
            "    dn = lax.conv_dimension_numbers(images.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))",
            "    out = lax.conv_general_dilated(images, kernel, window_strides=(1, 1), padding=padding.upper(), dimension_numbers=dn, feature_group_count=C)",
            '    if data_format == "channels_first":',
            "        out = jnp.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def jax_median_filter(images, kernel_size, padding, data_format):",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = images[None, ...]",
            '    if data_format == "channels_first":',
            "        images = jnp.transpose(images, (0, 2, 3, 1))",
            "    import jax.lax as lax",
            "    B, H, W, C = images.shape",
            "    ky, kx = kernel_size",
            "    if padding == 'same':",
            "        pad_y, pad_x = ky // 2, kx // 2",
            "        images = jnp.pad(images, ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)))",
            "        H, W = images.shape[1], images.shape[2]",
            "    out_H, out_W = H - ky + 1, W - kx + 1",
            "    patches = jax.lax.conv_general_dilated_patches(images, (ky, kx), (1, 1), 'VALID', dimension_numbers=('NHWC', 'OIHW', 'NHWC'))",
            "    patches = patches.reshape(B, out_H, out_W, ky * kx, C)",
            "    out = jnp.median(patches, axis=3)",
            '    if data_format == "channels_first":',
            "        out = jnp.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "def jax_extract_bounding_boxes(images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format):",
            "    import jax",
            "    import jax.numpy as jnp",
            '    if data_format == "channels_first":',
            "        images = jnp.transpose(images, (0, 2, 3, 1))",
            "    import jax.image as jimg",
            "    # JAX doesn't have a direct crop_and_resize equivalent natively exposed like tf.image.crop_and_resize",
            "    # We'll use the eager fallback mapped into JAX.",
            "    from ml_switcheroo_compiler.backends.eager_utils import extract_bounding_boxes_eager",
            "    return extract_bounding_boxes_eager(jnp, images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format)",
            "def jax_iou(boxes1, boxes2, bounding_box_format):",
            "    from ml_switcheroo_compiler.backends.eager_utils import iou_eager",
            "    import jax.numpy as jnp",
            "    return iou_eager(jnp, boxes1, boxes2, bounding_box_format)",
            "def jax_nms(boxes, scores, max_output_size, iou_threshold, score_threshold):",
            "    from ml_switcheroo_compiler.backends.eager_utils import nms_eager",
            "    import jax.numpy as jnp",
            "    return nms_eager(jnp, boxes, scores, max_output_size, iou_threshold, score_threshold)",
            "def jax_resize(images, size, interpolation, align_corners):",
            "    import jax.image as jimg",
            "    method = 'lanczos3' if interpolation == 'lanczos3' else 'bicubic'",
            "    return jimg.resize(images, (images.shape[0], size[0], size[1], images.shape[3]), method)",
            "def jax_istft(stft_tensor, frame_length, frame_step, fft_length, window, center):",
            "    from ml_switcheroo_compiler.backends.eager_utils import istft_eager",
            "    import jax.numpy as jnp",
            "    return istft_eager(jnp, stft_tensor, frame_length, frame_step, fft_length, window, center)",
            "def jax_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):",
            "    from ml_switcheroo_compiler.backends.eager_utils import mel_filterbank_eager",
            "    import jax.numpy as jnp",
            "    return mel_filterbank_eager(jnp, None, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)",
            "def jax_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):",
            "    from ml_switcheroo_compiler.backends.eager_utils import mfcc_eager",
            "    import jax.numpy as jnp",
            "    return mfcc_eager(jnp, spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs)",
            "def jax_perspective_transform(images, start_points, end_points, interpolation, fill_value, data_format):",
            "    def get_h(src, dst):",
            "        A = jnp.zeros((*dst.shape[:-2], 8, 8), dtype=jnp.float32)",
            "        B = jnp.zeros((*dst.shape[:-2], 8), dtype=jnp.float32)",
            "        for i in range(4):",
            "            u, v = dst[..., i, 0], dst[..., i, 1]",
            "            x, y = src[..., i, 0], src[..., i, 1]",
            "            A = A.at[..., i*2, 0].set(u)",
            "            A = A.at[..., i*2, 1].set(v)",
            "            A = A.at[..., i*2, 2].set(1.0)",
            "            A = A.at[..., i*2, 6].set(-x * u)",
            "            A = A.at[..., i*2, 7].set(-x * v)",
            "            A = A.at[..., i*2+1, 3].set(u)",
            "            A = A.at[..., i*2+1, 4].set(v)",
            "            A = A.at[..., i*2+1, 5].set(1.0)",
            "            A = A.at[..., i*2+1, 6].set(-y * u)",
            "            A = A.at[..., i*2+1, 7].set(-y * v)",
            "            B = B.at[..., i*2].set(x)",
            "            B = B.at[..., i*2+1].set(y)",
            "        h = jnp.linalg.solve(A, B)",
            "        return jnp.concatenate([h, jnp.ones((*dst.shape[:-2], 1), dtype=jnp.float32)], axis=-1).reshape((*dst.shape[:-2], 3, 3))",
            "    has_batch = images.ndim == 4",
            "    if not has_batch:",
            "        images = images[None, ...]",
            "        start_points = start_points[None, ...]",
            "        end_points = end_points[None, ...]",
            '    if data_format == "channels_first":',
            "        images = jnp.transpose(images, (0, 2, 3, 1))",
            "    H_mat = get_h(start_points, end_points)",
            "    B_sz, H_dim, W_dim, C_dim = images.shape",
            "    y_grid, x_grid = jnp.meshgrid(jnp.arange(H_dim), jnp.arange(W_dim), indexing='ij')",
            "    y_grid = y_grid.astype(jnp.float32)",
            "    x_grid = x_grid.astype(jnp.float32)",
            "    coords = jnp.stack([x_grid, y_grid, jnp.ones_like(x_grid)], axis=-1)",
            "    import jax",
            "    import jax.scipy.ndimage as ndimage",
            "    def process_batch(img, h_mat):",
            "        t_coords = coords @ h_mat.T",
            "        t_coords = t_coords / t_coords[..., 2:3]",
            "        src_x = t_coords[..., 0]",
            "        src_y = t_coords[..., 1]",
            "        def process_channel(c_img):",
            '            order = 1 if interpolation == "bilinear" else 0',
            "            return ndimage.map_coordinates(c_img, [src_y, src_x], order=order, mode='constant', cval=fill_value)",
            "        return jax.vmap(process_channel, in_axes=-1, out_axes=-1)(img)",
            "    out = jax.vmap(process_batch)(images, H_mat)",
            '    if data_format == "channels_first":',
            "        out = jnp.transpose(out, (0, 3, 1, 2))",
            "    if not has_batch:",
            "        out = out[0]",
            "    return out",
            "",
            "def jax_power_iteration(w, num_iters, u=None):",
            "    import jax",
            "    import jax.numpy as jnp",
            "    if u is None:",
            "        u = jnp.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)",
            "    def cond_fun(val):",
            "        return val[0] < num_iters",
            "    def body_fun(val):",
            "        i, u_curr, _ = val",
            "        w_t = jnp.swapaxes(w, -1, -2)",
            "        v_next = jnp.matmul(w_t, u_curr)",
            "        v_next = v_next / (jnp.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)",
            "        u_next = jnp.matmul(w, v_next)",
            "        u_next = u_next / (jnp.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)",
            "        return i + 1, u_next, v_next",
            "    init_v = jnp.zeros(w.shape[:-2] + (w.shape[-1], 1), dtype=w.dtype)",
            "    _, u_final, v_final = jax.lax.while_loop(cond_fun, body_fun, (0, u, init_v))",
            "    sigma = jnp.matmul(jnp.swapaxes(u_final, -1, -2), jnp.matmul(w, v_final))",
            "    return jnp.squeeze(v_final, -1), jnp.squeeze(u_final, -1), jnp.squeeze(jnp.squeeze(sigma, -1), -1)",
        ]

        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
