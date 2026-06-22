# ruff: noqa: E402, D100, D101
from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_stft_attributes,
    extract_mel_attributes,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)

"""NumPy code generator and eager execution backend."""

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


from ml_switcheroo_compiler.backends.common.generator_mixins import (
    SharedASTGeneratorMixin,
    GroupNormConfig,
)


@register_backend("numpy")
class NumpyGenerator(SharedASTGeneratorMixin, PythonStringGenerator):
    """Generates NumPy python code from IR."""

    def _get_backend_prefix(self) -> str:
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()
        res.extend(
            self._get_group_norm_code(
                GroupNormConfig(
                    "np",
                    "numpy as np",
                    "np.reshape",
                    "np.mean",
                    "np.var",
                    "np.sqrt",
                    dim_arg="axis",
                    keepdim_arg="keepdims",
                )
            )
        )
        return res

    _import_header = (
        "import numpy as np",
        "import scipy.special",
        "",
        "def np_perspective_transform(images, start_points, end_points, interpolation, fill_value, data_format):",
        "    def get_h(src, dst):",
        "        A = np.zeros((*dst.shape[:-2], 8, 8), dtype=np.float32)",
        "        B = np.zeros((*dst.shape[:-2], 8), dtype=np.float32)",
        "        for i in range(4):",
        "            u, v = dst[..., i, 0], dst[..., i, 1]",
        "            x, y = src[..., i, 0], src[..., i, 1]",
        "            A[..., i*2, 0] = u",
        "            A[..., i*2, 1] = v",
        "            A[..., i*2, 2] = 1.0",
        "            A[..., i*2, 6] = -x * u",
        "            A[..., i*2, 7] = -x * v",
        "            A[..., i*2+1, 3] = u",
        "            A[..., i*2+1, 4] = v",
        "            A[..., i*2+1, 5] = 1.0",
        "            A[..., i*2+1, 6] = -y * u",
        "            A[..., i*2+1, 7] = -y * v",
        "            B[..., i*2] = x",
        "            B[..., i*2+1] = y",
        "        h = np.linalg.solve(A, B)",
        "        return np.reshape(np.concatenate([h, np.ones((*dst.shape[:-2], 1), dtype=np.float32)], axis=-1), (*dst.shape[:-2], 3, 3))",
        "    has_batch = images.ndim == 4",
        "    if not has_batch:",
        "        images = np.expand_dims(images, 0)",
        "        start_points = np.expand_dims(start_points, 0)",
        "        end_points = np.expand_dims(end_points, 0)",
        '    if data_format == "channels_first":',
        "        images = np.transpose(images, (0, 2, 3, 1))",
        "    H_mat = get_h(start_points, end_points)",
        "    B_sz, H_dim, W_dim, C_dim = images.shape",
        "    y_grid, x_grid = np.meshgrid(np.arange(H_dim), np.arange(W_dim), indexing='ij')",
        "    y_grid = y_grid.astype(np.float32)",
        "    x_grid = x_grid.astype(np.float32)",
        "    coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)",
        "    ",
        "    out_list = []",
        "    for b in range(B_sz):",
        "        t_coords = np.matmul(coords, np.transpose(H_mat[b]))",
        "        t_coords = t_coords / t_coords[..., 2:3]",
        "        src_x = t_coords[..., 0]",
        "        src_y = t_coords[..., 1]",
        "        c_list = []",
        "        for c in range(C_dim):",
        '            order = 1 if interpolation == "bilinear" else 0',
        "            from scipy.ndimage import map_coordinates",
        "            c_res = map_coordinates(images[b, ..., c], [src_y, src_x], order=order, mode='constant', cval=fill_value)",
        "            c_list.append(c_res)",
        "        out_list.append(np.stack(c_list, axis=-1))",
        "    out = np.stack(out_list, axis=0)",
        "    ",
        '    if data_format == "channels_first":',
        "        out = np.transpose(out, (0, 3, 1, 2))",
        "    if not has_batch:",
        "        out = out[0]",
        "    return out",
        "",
        "def np_power_iteration(w, num_iters, u=None):",
        "    if u is None:",
        "        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)",
        "    for _ in range(num_iters):",
        "        w_t = np.swapaxes(w, -1, -2)",
        "        v = np.matmul(w_t, u)",
        "        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)",
        "        u = np.matmul(w, v)",
        "        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)",
        "    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))",
        "    return np.squeeze(v, -1), np.squeeze(u, -1), np.squeeze(np.squeeze(sigma, -1), -1)",
    )

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"np_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_TensorScatterUpdate(
        self, node: IRNode, input_vars: list[str], **kwargs: object
    ) -> str:
        """Handle TensorScatterUpdate."""
        return f"(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterAdd."""
        return f"(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMax(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterMax."""
        return f"(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterMin."""
        return f"(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"np.einsum('{eq}', {args_str})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback visit."""
        op_type = node.op_type
        op_map = {
            "Add": "np.add",
            "Zeros": "np.zeros",
            "Ones": "np.ones",
            "Full": "np.full",
            "Arange": "np.arange",
            "Sort": "np.sort",
            "ArgSort": "np.argsort",
            "Allclose": "np.allclose",
            "Fft": "np.fft.fft",
            "Rfft": "np.fft.rfft",
            "Fftn": "np.fft.fftn",
            "Erfinv": "scipy.special.erfinv",
            "NanToNum": "np.nan_to_num",
            "Subtract": "np.subtract",
            "Multiply": "np.multiply",
            "TrueDivide": "np.divide",
            "Exp": "np.exp",
            "Log": "np.log",
            "Matmul": "np.matmul",
            "Sin": "np.sin",
            "Cos": "np.cos",
            "Sum": "np.sum",
            "Mean": "np.mean",
            "Max": "np.max",
            "Min": "np.min",
            "BroadcastTo": "np.broadcast_to",
            "Reshape": "np.reshape",
            "Transpose": "np.transpose",
            "Equal": "np.equal",
            "NotEqual": "np.not_equal",
            "Greater": "np.greater",
            "Less": "np.less",
            "Negative": "np.negative",
        }

        np_func = op_map.get(op_type, f"np.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

        return f"{np_func}({args_str})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (
            _extract_extract_boxes_attributes(node)
        )
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"np_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"np_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"np_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"np_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"np_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, _ = (
            extract_mel_attributes(node)
        )
        return f"np_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        num_mel_bins, _, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs = (
            extract_mel_attributes(node)
        )
        return f"np_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"
