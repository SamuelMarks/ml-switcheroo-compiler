# ruff: noqa: E402, D100, D101
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)

"""MLX Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import (
    SharedASTGeneratorMixin,
    GroupNormConfig,
)
from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.registry import register_backend


class MLXNNOpsVisitorMixin:
    """MLX NN ops visitor mixin."""

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"mlx_power_iteration({input_vars[0]}, {num_iters}, {u_var})"


class MLXVisionVisitorMixin:
    """MLX Vision ops visitor mixin."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (
            _extract_extract_boxes_attributes(node)
        )
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

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"


class MLXAudioVisitorMixin:
    """MLX Audio ops visitor mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        from ml_switcheroo_compiler.backends.common.audio_utils import extract_stft_attributes

        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
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


@register_backend("mlx")
class MLXCodeGenerator(
    SharedASTGeneratorMixin,
    ClassBasedGenerator,
    MLXNNOpsVisitorMixin,
    MLXVisionVisitorMixin,
    MLXAudioVisitorMixin,
):
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

    _forward_method_name = "__call__"

    def _get_prefix_code(self) -> list[str]:
        res = [
            "import mlx.core as mx",
            "import mlx.nn as nn\n",
            "def mlx_resize(images, size, interpolation, align_corners):",
            "    import mlx.core as mx",
            "    from ml_switcheroo_compiler.backends.eager.vision_geometric import resize_eager",
            "    from ml_switcheroo_compiler.ops.configs import ResizeOptions",
            "    # Fallback to eager numpy for bicubic/lanczos3 on mlx",
            "    import numpy as np",
            "    imgs_np = np.array(images)",
            "    config = ResizeOptions(interpolation=interpolation, align_corners=align_corners)",
            "    out = resize_eager(np, imgs_np, size, config)",
            "    return mx.array(out)",
            "def mlx_istft(matrix, frame_length, frame_step, fft_length, window, center):",
            "    import mlx.core as mx",
            "    import numpy as np",
            "    from ml_switcheroo_compiler.backends.eager.audio import istft_eager",
            "    from ml_switcheroo_compiler.ops.configs import STFTConfig",
            "    config = STFTConfig(frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, window=window, center=center)",
            "    out = istft_eager(np, np.array(matrix), config)",
            "    return mx.array(out)",
            "def mlx_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):",
            "    import mlx.core as mx",
            "    import numpy as np",
            "    from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager",
            "    from ml_switcheroo_compiler.ops.configs import MelConfig",
            "    config = MelConfig(num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=sample_rate, lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz)",
            "    out = mel_filterbank_eager(np, config)",
            "    return mx.array(out)",
            "def mlx_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):",
            "    import mlx.core as mx",
            "    import numpy as np",
            "    from ml_switcheroo_compiler.backends.eager.audio import mfcc_eager",
            "    from ml_switcheroo_compiler.ops.configs import MelConfig",
            "    config = MelConfig(num_mel_bins=num_mel_bins, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=sample_rate, lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz, num_mfccs=num_mfccs)",
            "    out = mfcc_eager(np, np.array(spectrogram), config)",
            "    return mx.array(out)",
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
        ]
        res.extend(
            self._get_group_norm_code(
                GroupNormConfig(
                    "mx",
                    "mlx.core as mx",
                    "mx.reshape",
                    "mx.mean",
                    "mx.var",
                    "mx.sqrt",
                    dim_arg="axis",
                    keepdim_arg="keepdims",
                )
            )
        )
        return res
