"""PyTorch Target Emission."""

from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("pytorch")
class PyTorchCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """PyTorch code generator."""

    def _get_backend_prefix(self) -> str:
        return "pt"

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        # In an actual implementation we would pre-allocate or use specific torch APIs
        return f"torch.distributed.all_gather_into_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        return f"torch.distributed.reduce_scatter_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        return f"torch.distributed.all_reduce({tensor})"

    """Emit PyTorch-compatible code from IR."""

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"pt_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"pt_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size = node.attributes.get("kernel_size")
        sigma = node.attributes.get("sigma")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"pt_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size = node.attributes.get("kernel_size")
        padding = node.attributes.get("padding", "same")
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"pt_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size = node.attributes.get("crop_size")
        interpolation = node.attributes.get("interpolation", "bilinear")
        extrapolation_value = node.attributes.get("extrapolation_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"pt_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"pt_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"pt_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"pt_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"pt_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"  # PyTorch doesn't natively support lanczos3 in F.interpolate, mapping to bicubic

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length = node.attributes.get("frame_length")
        frame_step = node.attributes.get("frame_step")
        fft_length = node.attributes.get("fft_length", None)
        window = node.attributes.get("window", "hann")
        center = node.attributes.get("center", True)
        fft_len_str = "None" if fft_length is None else str(fft_length)
        return f"pt_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins = node.attributes.get("num_mel_bins")
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
        sample_rate = node.attributes.get("sample_rate")
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")
        return f"pt_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        sample_rate = node.attributes.get("sample_rate")
        num_mel_bins = node.attributes.get("num_mel_bins", 40)
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
        num_mfccs = node.attributes.get("num_mfccs", 13)
        return f"pt_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation = node.attributes.get("interpolation", "bilinear")
        fill_value = node.attributes.get("fill_value", 0.0)
        data_format = node.attributes.get("data_format", None)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"pt_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"torch.einsum('{eq}', {args_str})"

    def visit_TensorScatterUpdate(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Handle TensorScatterUpdate nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterAdd nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]}, accumulate=True)"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated PyTorch Python code
        """
        op_type = getattr(node, "op_type", "")

        if op_type == "TensorScatterMax":
            return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"
        if op_type == "TensorScatterMin":
            return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

        if op_type == "Einsum":
            args_str = ", ".join(input_vars)
            eq = kwargs.get("equation", "")
            return f"torch.einsum('{eq}', {args_str})"

        ops_map = self._get_ops_map(kwargs)

        if op_type in ops_map:
            return self._format_mapped_op(ops_map[op_type], input_vars, kwargs)

        return self.format_generic_fallback(op_type, input_vars, kwargs)

    def _get_ops_map(self, kwargs: dict) -> dict:
        """Execute _get_ops_map.

        Args:
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        return {
            "Matmul": "torch.matmul({0}, {1})",
            "Dot": "torch.dot({0}, {1})",
            "BroadcastTo": "{0}.expand({shape})",
            "Reshape": "torch.reshape({0}, {shape})",
            "TrueDivide": "torch.true_divide({0}, {1})",
            "Arange": "torch.arange({0})",
            "Zeros": "torch.zeros({shape})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
            "Ones": "torch.ones({shape})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
            "Full": "torch.full({shape}, {fill_value})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
            "Sort": "torch.sort({0}, axis={dimension})",
            "ArgSort": "torch.argsort({0}, axis={dimension})",
            "Allclose": "torch.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "torch.fft.fft({0})",
            "Rfft": "torch.fft.rfft({0})",
            "Fftn": "torch.fft.fftn({0})",
            "Erfinv": "torch.erfinv({0})",
            "NanToNum": "torch.nan_to_num({0}, nan={nan}, posinf={posinf}, neginf={neginf})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "TensorScatterUpdate": "{0}.clone().index_put_(tuple({1}.unbind(-1)), {2})",
            "TensorScatterAdd": "{0}.clone().index_put_(tuple({1}.unbind(-1)), {2}, accumulate=True)",
            "Transpose": "torch.permute({0}, {axes})" if "axes" in kwargs else "{0}.t()",
            "Sum": "torch.sum({0}, dim={axis}, keepdim={keepdims})",
            "Mean": "torch.mean({0}, dim={axis}, keepdim={keepdims})",
            "Max": "torch.max({0}, dim={axis}, keepdim={keepdims})",
        }

    def _format_mapped_op(self, fmt: str, input_vars: list[str], kwargs: dict) -> str:
        """Execute _format_mapped_op.

        Args:
            fmt (Any): Argument fmt.
            input_vars (Any): Argument input_vars.
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        # Replace kwargs placeholders
        for k, v in kwargs.items():
            if f"{{{k}}}" in fmt:
                fmt = fmt.replace(f"{{{k}}}", str(v))
        # Special case for keepdims
        if "keepdims" in fmt and "keepdims" not in kwargs:
            fmt = fmt.replace("keepdim={keepdims}", "keepdim=False")
        if "axis" in fmt and "axis" not in kwargs:
            fmt = fmt.replace(", dim={axis}", "")
        # Replace args placeholders
        for i, var in enumerate(input_vars):
            fmt = fmt.replace(f"{{{i}}}", var)
        return fmt

    def format_generic_fallback(self, op_type: str, input_vars: list[str], kwargs: dict) -> str:
        """Execute _format_generic_fallback.

        Args:
            op_type (Any): Argument op_type.
            input_vars (Any): Argument input_vars.
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext("torch", op_type, input_vars, kwargs, "dim", "keepdim")
        )

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = self.{var_name}")

    def generate(self) -> str:
        """Generate PyTorch model code from the IR graph.

        Returns:
            str: The generated PyTorch Python code
        """
        self.code = [
            self.header.strip(),
            "import torch",
            "import torch.nn as nn\n",
            "def pt_group_norm(x, groups, weight=None, bias=None, axis=-1, epsilon=1e-5):",
            "    import torch",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = torch.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    mean = torch.mean(reshaped_x, dim=reduction_axes, keepdim=True)",
            "    var = torch.var(reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False)",
            "    normalized = (reshaped_x - mean) / torch.sqrt(var + epsilon)",
            "    out = torch.reshape(normalized, shape)",
            "    if weight is not None:",
            "        w_shape = [1] * ndims",
            "        w_shape[axis] = C",
            "        weight = torch.reshape(weight, w_shape)",
            "        out = out * weight",
            "    if bias is not None:",
            "        b_shape = [1] * ndims",
            "        b_shape[axis] = C",
            "        bias = torch.reshape(bias, b_shape)",
            "        out = out + bias",
            "    return out",
            "def pt_group_mean(x, groups, axis=-1):",
            "    import torch",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = torch.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return torch.mean(reshaped_x, dim=reduction_axes, keepdim=True)",
            "def pt_group_variance(x, groups, axis=-1):",
            "    import torch",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            "    reshaped_x = torch.reshape(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            "    return torch.var(reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False)",
            "def pt_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):",
            "    import torchaudio",
            "    return torchaudio.functional.melscale_fbanks(",
            "        num_spectrogram_bins, lower_edge_hertz, upper_edge_hertz, num_mel_bins, sample_rate, 'htk')",
            "def pt_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):",
            "    import torchaudio",
            "    import torch",
            "    mel_spec = torch.matmul(spectrogram, pt_mel_filterbank(num_mel_bins, spectrogram.shape[-1], sample_rate, lower_edge_hertz, upper_edge_hertz).to(spectrogram.device))",
            "    log_mel = torch.log(mel_spec + 1e-6)",
            "    return torchaudio.functional.create_dct(num_mfccs, num_mel_bins, 'ortho').to(spectrogram.device).matmul(log_mel.unsqueeze(-2)).squeeze(-2)",  # PyTorch DCT matrix multiplication
            "def pt_istft(stft_tensor, frame_length, frame_step, fft_length, window, center):",
            "    import torch",
            "    if fft_length is None: fft_length = frame_length",
            "    if window == 'hann': win = torch.hann_window(frame_length, periodic=True, device=stft_tensor.device)",
            "    elif window == 'hamming': win = torch.hamming_window(frame_length, periodic=True, device=stft_tensor.device)",
            "    else: win = torch.ones(frame_length, device=stft_tensor.device)",
            "    return torch.istft(stft_tensor, n_fft=fft_length, hop_length=frame_step, win_length=frame_length, window=win, center=center, normalized=False, return_complex=False)",
            "def pt_resize(images, size, interpolation, align_corners):",
            "    import torch.nn.functional as F",
            "    images = images.permute(0, 3, 1, 2)",
            "    out = F.interpolate(images, size=size, mode=interpolation, align_corners=align_corners)",
            "    out = out.permute(0, 2, 3, 1)",
            "    return out",
            "def pt_iou(boxes1, boxes2, bounding_box_format):",
            "    import torchvision.ops as tv_ops",
            "    def to_xyxy(boxes, fmt):",
            "        if format == 'yxyx': return boxes[:, [1, 0, 3, 2]]",
            "        if format == 'xywh':",
            "            x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]",
            "            return torch.stack([x, y, x + w, y + h], dim=-1)",
            "        if format == 'center_xywh':",
            "            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]",
            "            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)",
            "        return boxes",
            "    b1 = to_xyxy(boxes1, bounding_box_format)",
            "    b2 = to_xyxy(boxes2, bounding_box_format)",
            "    return tv_ops.box_iou(b1, b2)",
            "def pt_nms(boxes, scores, max_output_size, iou_threshold, score_threshold):",
            "    import torchvision.ops as tv_ops",
            "    mask = scores > score_threshold",
            "    f_boxes = boxes[mask]",
            "    f_scores = scores[mask]",
            "    orig_idx = torch.arange(len(scores), device=scores.device)[mask]",
            "    keep = tv_ops.nms(f_boxes, f_scores, iou_threshold)",
            "    keep = keep[:max_output_size]",
            "    return orig_idx[keep].to(torch.int32)",
            "def pt_extract_bounding_boxes(images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format):",
            "    import torchvision.ops as tv_ops",
            '    if data_format != "channels_first":',
            "        images = images.permute(0, 3, 1, 2)",
            "    N = boxes.shape[0]",
            "    H, W = images.shape[2], images.shape[3]",
            "    boxes_abs = boxes.clone()",
            "    boxes_abs[:, 0] = boxes[:, 0] * (H - 1)",
            "    boxes_abs[:, 1] = boxes[:, 1] * (W - 1)",
            "    boxes_abs[:, 2] = boxes[:, 2] * (H - 1)",
            "    boxes_abs[:, 3] = boxes[:, 3] * (W - 1)",
            "    boxes_abs = boxes_abs[:, [1, 0, 3, 2]]",  # Convert y1x1y2x2 to x1y1x2y2 for torchvision
            "    rois = torch.cat([box_indices.unsqueeze(1).to(boxes_abs.dtype), boxes_abs], dim=1)",
            "    out = tv_ops.roi_align(images, rois, output_size=crop_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True)",
            '    if data_format != "channels_first":',
            "        out = out.permute(0, 2, 3, 1)",
            "    return out",
            "def pt_median_filter(images, kernel_size, padding, data_format):",
            "    import torch.nn.functional as F",
            "    import torch",
            "    has_batch = images.dim() == 4",
            "    if not has_batch:",
            "        images = images.unsqueeze(0)",
            '    if data_format != "channels_first":',
            "        images = images.permute(0, 3, 1, 2)",
            "    B, C, H, W = images.shape",
            "    ky, kx = kernel_size",
            "    if padding == 'same':",
            "        pad_y = ky // 2",
            "        pad_x = kx // 2",
            "        images = F.pad(images, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0.0)",
            "    out = F.unfold(images, kernel_size=(ky, kx))",
            "    out = out.view(B, C, ky * kx, -1)",
            "    out = out.median(dim=2).values",
            "    out = out.view(B, C, images.shape[2] - ky + 1, images.shape[3] - kx + 1)",
            '    if data_format != "channels_first":',
            "        out = out.permute(0, 2, 3, 1)",
            "    if not has_batch:",
            "        out = out.squeeze(0)",
            "    return out",
            "def pt_gaussian_blur(images, kernel_size, sigma, padding, data_format):",
            "    import torch.nn.functional as F",
            "    import torch",
            "    has_batch = images.dim() == 4",
            "    if not has_batch:",
            "        images = images.unsqueeze(0)",
            '    if data_format != "channels_first":',
            "        images = images.permute(0, 3, 1, 2)",
            "    B, C, H, W = images.shape",
            "    ky, kx = kernel_size",
            "    sy, sx = sigma",
            "    y = torch.arange(-ky // 2 + 1, ky // 2 + 1, device=images.device, dtype=images.dtype)",
            "    x = torch.arange(-kx // 2 + 1, kx // 2 + 1, device=images.device, dtype=images.dtype)",
            "    yy, xx = torch.meshgrid(y, x, indexing='ij')",
            "    kernel = torch.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))",
            "    kernel = kernel / torch.sum(kernel)",
            "    kernel = kernel.view(1, 1, ky, kx).expand(C, 1, ky, kx)",
            "    if padding == 'same':",
            "        pad_y = ky // 2",
            "        pad_x = kx // 2",
            "        images = F.pad(images, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0.0)",
            "    out = F.conv2d(images, kernel, groups=C)",
            '    if data_format != "channels_first":',
            "        out = out.permute(0, 2, 3, 1)",
            "    if not has_batch:",
            "        out = out.squeeze(0)",
            "    return out",
            "def pt_elastic_transform(images, displacement, interpolation, fill_value, data_format):",
            "    import torch.nn.functional as F",
            "    has_batch = images.dim() == 4",
            "    if not has_batch:",
            "        images = images.unsqueeze(0)",
            "        displacement = displacement.unsqueeze(0)",
            '    if data_format != "channels_first":',
            "        images = images.permute(0, 3, 1, 2)",
            "    B, C, H, W = images.shape",
            "    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=images.device), torch.arange(W, device=images.device), indexing='ij')",
            "    grid_y = grid_y.expand(B, -1, -1) + displacement[..., 0]",
            "    grid_x = grid_x.expand(B, -1, -1) + displacement[..., 1]",
            "    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0",
            "    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0",
            "    grid = torch.stack([grid_x, grid_y], dim=-1)",
            "    out = F.grid_sample(images, grid, mode=interpolation, padding_mode='zeros', align_corners=True)",
            '    if data_format != "channels_first":',
            "        out = out.permute(0, 2, 3, 1)",
            "    if not has_batch:",
            "        out = out.squeeze(0)",
            "    return out",
            "def pt_power_iteration(w, num_iters, u=None):",
            "    import torch",
            "    if u is None:",
            "        u = torch.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype, device=w.device)",
            "    for _ in range(num_iters):",
            "        w_t = w.transpose(-1, -2)",
            "        v = torch.matmul(w_t, u)",
            "        v = v / (torch.linalg.norm(v, dim=-2, keepdim=True) + 1e-12)",
            "        u = torch.matmul(w, v)",
            "        u = u / (torch.linalg.norm(u, dim=-2, keepdim=True) + 1e-12)",
            "    sigma = torch.matmul(u.transpose(-1, -2), torch.matmul(w, v))",
            "    return v.squeeze(-1), u.squeeze(-1), sigma.squeeze(-1).squeeze(-1)",
            "class CompiledModel(nn.Module):",
        ]

        # __init__
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")

        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(
                    f"self.register_parameter('{var_name}', "
                    f"nn.Parameter(torch.tensor({val_repr})))",
                )
                has_params = True

        if not has_params:
            self.add_line("pass")

        self.add_line("")
        self.indent_level -= 1

        # forward
        self.add_line("def forward(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
