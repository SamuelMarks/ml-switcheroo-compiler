"""Module docstring."""

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig
from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)
from ml_switcheroo_compiler.backends.registry import register_backend

_MLX_RESIZE_TMPL = """\
def mlx_resize(images, size, interpolation, align_corners):
    import mlx.core as mx
    from ml_switcheroo_compiler.backends.eager.vision_geometric import resize_eager
    from ml_switcheroo_compiler.ops.configs import ResizeOptions
    # Fallback to eager numpy for bicubic/lanczos3 on mlx
    import numpy as np
    imgs_np = np.array(images)
    config = ResizeOptions(interpolation=interpolation, align_corners=align_corners)
    out = resize_eager(np, imgs_np, size, config)
    return mx.array(out)"""

_MLX_ISTFT_TMPL = """\
def mlx_istft(matrix, config: STFTConfig):
    import mlx.core as mx
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.audio import istft_eager
    from ml_switcheroo_compiler.ops.configs import STFTConfig
    config = STFTConfig(frame_length=config.frame_length, frame_step=config.frame_step, fft_length=config.fft_length, window=config.window, center=config.center)
    out = istft_eager(np, np.array(matrix), config)
    return mx.array(out)"""

_MLX_MEL_FILTERBANK_TMPL = """\
def mlx_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):
    import mlx.core as mx
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager
    from ml_switcheroo_compiler.ops.configs import MelConfig
    config = MelConfig(num_mel_bins=config.num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=config.sample_rate, lower_edge_hertz=config.lower_edge_hertz, upper_edge_hertz=config.upper_edge_hertz)
    out = mel_filterbank_eager(np, config)
    return mx.array(out)"""

_MLX_MFCC_TMPL = """\
def mlx_mfcc(spectrogram, config: MFCCConfig):
    import mlx.core as mx
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.audio import mfcc_eager
    from ml_switcheroo_compiler.ops.configs import MelConfig
    config = MelConfig(num_mel_bins=config.num_mel_bins, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=config.sample_rate, lower_edge_hertz=config.lower_edge_hertz, upper_edge_hertz=config.upper_edge_hertz, num_mfccs=config.num_mfccs)
    out = mfcc_eager(np, np.array(spectrogram), config)
    return mx.array(out)"""

_MLX_POWER_ITERATION_TMPL = """\
def mlx_power_iteration(w, num_iters, u=None):
    import mlx.core as mx
    if u is None:
        u = mx.ones(w.shape[:-2] + [w.shape[-2], 1], dtype=w.dtype)
    def body_fn(val):
        i, u_curr, _ = val
        w_t = mx.swapaxes(w, -1, -2)
        v_next = mx.matmul(w_t, u_curr)
        v_next = v_next / (mx.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)
        u_next = mx.matmul(w, v_next)
        u_next = u_next / (mx.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)
        return i + 1, u_next, v_next
    def cond_fn(val):
        return val[0] < num_iters
    init_v = mx.zeros(w.shape[:-2] + [w.shape[-1], 1], dtype=w.dtype)
    _, u_final, v_final = mx.while_loop(cond_fn, body_fn)( (mx.array(0), u, init_v) )
    sigma = mx.matmul(mx.swapaxes(u_final, -1, -2), mx.matmul(w, v_final))
    return mx.squeeze(v_final, -1), mx.squeeze(u_final, -1), mx.squeeze(mx.squeeze(sigma, -1), -1)"""


class MLXNNOpsVisitor:
    """MLX NN ops visitor mixin."""

    def visit_Rope(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Rope."""
        dim = node.attributes.get("dim")
        traditional = node.attributes.get("traditional", False)
        base = node.attributes.get("base", 10000.0)
        scale = node.attributes.get("scale", 1.0)
        offset = node.attributes.get("offset", 0)
        return f"mx.fast.rope({input_vars[0]}, {dim}, traditional={traditional}, base={base}, scale={scale}, offset={offset})"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"mlx_power_iteration({input_vars[0]}, {num_iters}, {u_var})"


class MLXVisionVisitor:
    """MLX Vision ops visitor mixin."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"mlx_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"mlx_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"  # pragma: no cover

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"mlx_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"  # pragma: no cover

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (  # pragma: no cover
            _extract_extract_boxes_attributes(node)
        )
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"mlx_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"  # pragma: no cover

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")  # pragma: no cover
        return f"mlx_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"  # pragma: no cover

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")  # pragma: no cover
        iou_threshold = node.attributes.get("iou_threshold", 0.5)  # pragma: no cover
        score_threshold = node.attributes.get("score_threshold", float("-inf"))  # pragma: no cover
        return f"mlx_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"  # pragma: no cover

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        # mlx has no bicubic resize. We fallback to map to eager utility
        return f"mlx_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"  # pragma: no cover

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        return f"mlx_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"  # pragma: no cover

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"mlx_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover


class MLXAudioVisitor:
    """MLX Audio ops visitor mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        from ml_switcheroo_compiler.backends.common.audio_utils import (
            extract_stft_attributes,
        )  # pragma: no cover

        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)  # pragma: no cover
        return f"mlx_istft({input_vars[0]}, STFTConfig({frame_length}, {frame_step}, {fft_len_str}, '{window}', {center}))"  # pragma: no cover

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins = node.attributes.get("num_mel_bins")  # pragma: no cover
        num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")  # pragma: no cover
        sample_rate = node.attributes.get("sample_rate")  # pragma: no cover
        lower_edge_hertz = node.attributes.get("lower_edge_hertz")  # pragma: no cover
        upper_edge_hertz = node.attributes.get("upper_edge_hertz")  # pragma: no cover
        return f"mlx_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"  # pragma: no cover

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        sample_rate = node.attributes.get("sample_rate")  # pragma: no cover
        num_mel_bins = node.attributes.get("num_mel_bins", 40)  # pragma: no cover
        lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)  # pragma: no cover
        upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)  # pragma: no cover
        num_mfccs = node.attributes.get("num_mfccs", 13)  # pragma: no cover
        return f"mlx_mfcc({input_vars[0]}, MFCCConfig({sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs}))"  # pragma: no cover


@register_backend("mlx")
class MLXCodeGenerator(
    ClassBasedGenerator,
):
    """Emit MLX-compatible code from IR."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend(
            [
                SharedASTGeneratorVisitor(generator=self),
                MLXNNOpsVisitor(),
                MLXVisionVisitor(),
                MLXAudioVisitor(),
            ]
        )

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "mlx"  # pragma: no cover

    _SIMPLE_OPS_MAP = {
        "Matmul": "mx.matmul({0}, {1})",
        "Trace": "tf.linalg.trace",
        "Adjoint": "tf.linalg.adjoint",
        "BandPart": "tf.linalg.band_part",
        "CholeskySolve": "tf.linalg.cholesky_solve",
        "Dot": "mx.dot({0}, {1})",
        "BroadcastTo": "mx.broadcast_to({0}, {shape})",
        "Reshape": "mx.reshape({0}, {shape})",
        "TruncateDiv": "mx.trunc(mx.divide({0}, {1}))",
        "TruncateMod": "mx.remainder({0}, {1})",
        "TrueDivide": "mx.divide({0}, {1})",
        "Arange": "mx.arange({0})",
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
        "Ifft": "mx.fft.ifft({0}, n={n}, axis={axis})",
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
        "Eigh": "mx.linalg.eigh({0})",
        "MatrixPower": "mx.linalg.matrix_power({0}, {n})",
        "Norm": "mx.linalg.norm({0}, ord={ord}, axis={axis}, keepdims={keepdims})",
        "Det": "mx.linalg.det({0})",
        "Slogdet": "mx.linalg.slogdet({0})",
        "Pinv": "mx.linalg.pinv({0})",
        "TriInv": "mx.linalg.inv({0})",
        "TriangularSolve": "mx.linalg.solve_triangular({0}, {1}, upper=not {lower})",
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

    def _format_einsum(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        """Function docstring.

        Args:
        input_vars: Arg.
        kwargs: Arg.
        """
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"mx.einsum('{eq}', {args_str})"  # pragma: no cover

    def _format_zeros_ones(self, op: str, kwargs: dict[str, object]) -> str:
        """Function docstring.

        Args:
        op: Arg.
        kwargs: Arg.
        """
        dtype_str = f", dtype=mx.{kwargs['dtype']}" if "dtype" in kwargs else ""
        return f"mx.{op.lower()}({{shape}}){dtype_str}"

    def _format_full(self, kwargs: dict[str, object]) -> str:
        """Function docstring.

        Args:
        kwargs: Arg.
        """
        dtype_str = f", dtype=mx.{kwargs['dtype']}" if "dtype" in kwargs else ""  # pragma: no cover
        val = "{fill_value}" if "fill_value" in kwargs else "{value}"  # pragma: no cover
        return f"mx.full({{shape}}, {val}){dtype_str}"  # pragma: no cover

    def _format_transpose(self, kwargs: dict[str, object]) -> str:
        """Function docstring.

        Args:
        kwargs: Arg.
        """
        return "mx.transpose({0}, {axes})" if "axes" in kwargs else "mx.transpose({0})"

    def _format_random_categorical(self, input_vars: list[str], kwargs: dict[str, object]) -> str:
        """Function docstring.

        Args:
        input_vars: Arg.
        kwargs: Arg.
        """
        if len(input_vars) > 1:  # pragma: no cover
            return "mx.random.categorical(logits={1}, axis={axis}, shape={shape}, key={0})"  # pragma: no cover
        return "mx.random.categorical(axis={axis}, shape={shape}, key={0})"  # pragma: no cover

    def visit_Concatenate(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover  # pragma: no cover
        # pragma: no cover(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Concatenate."""
        axis = kwargs.get("axis", 0)
        vars_str = ", ".join(input_vars)
        return f"mx.concatenate([{vars_str}], axis={axis})"

    def visit_Stack(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Stack."""
        axis = kwargs.get("axis", 0)
        vars_str = ", ".join(input_vars)
        return f"mx.stack([{vars_str}], axis={axis})"

    def visit_Partition(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Partition."""
        axis = kwargs.get("axis", -1)
        kth = input_vars[1]
        return f"mx.partition({input_vars[0]}, {kth} if isinstance({kth}, int) else {kth}[0], axis={axis})"

    def visit_Argpartition(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Argpartition."""
        axis = kwargs.get("axis", -1)
        kth = input_vars[1]
        return f"mx.argpartition({input_vars[0]}, {kth} if isinstance({kth}, int) else {kth}[0], axis={axis})"

    def visit_Repeat(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Repeat."""
        repeats = kwargs.get("repeats")
        axis = kwargs.get("axis", None)
        axis_str = f", axis={axis}" if axis is not None else ""
        return f"mx.repeat({input_vars[0]}, {repeats}{axis_str})"

    def visit_Roll(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Roll."""
        shift = kwargs.get("shift", kwargs.get("shifts"))
        axis = kwargs.get("axis", kwargs.get("dims", None))
        axis_str = f", axis={axis}" if axis is not None else ""
        return f"mx.roll({input_vars[0]}, {shift}{axis_str})"

    def visit_Tile(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Tile."""
        reps = kwargs.get("reps")
        return f"mx.tile({input_vars[0]}, {reps})"

    def visit_TopK(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle TopK."""
        k = kwargs.get("k")
        return_indices = kwargs.get("return_indices", False)
        var = input_vars[0]
        if return_indices:
            return f"mx.argsort({var}, axis=-1)[..., -({k}):][..., ::-1]"
        else:
            return f"mx.take_along_axis({var}, mx.argsort({var}, axis=-1)[..., -({k}):][..., ::-1], axis=-1)"

    def visit_Moveaxis(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle Moveaxis."""
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        return f"mx.moveaxis({input_vars[0]}, {source}, {destination})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"mlx_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_NanToNum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle NanToNum."""
        args = [input_vars[0]]  # pragma: no cover
        if "nan" in kwargs:  # pragma: no cover
            args.append(f"nan={kwargs['nan']}")  # pragma: no cover
        if "posinf" in kwargs:  # pragma: no cover
            args.append(f"posinf={kwargs['posinf']}")  # pragma: no cover
        if "neginf" in kwargs:  # pragma: no cover
            args.append(f"neginf={kwargs['neginf']}")  # pragma: no cover
        return f"mx.nan_to_num({', '.join(args)})"  # pragma: no cover

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum."""
        return self._format_einsum(input_vars, kwargs)  # pragma: no cover

    def visit_Zeros(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Zeros."""
        fmt = self._format_zeros_ones("Zeros", kwargs)
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

    def visit_Ones(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Ones."""
        fmt = self._format_zeros_ones("Ones", kwargs)  # pragma: no cover
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)  # pragma: no cover

    def visit_Full(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Full."""
        fmt = self._format_full(kwargs)  # pragma: no cover
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)  # pragma: no cover

    def visit_ConstantOfShape(self, node: object, input_vars: list[str], **kwargs: object) -> str:  # pragma: no cover
        """Handle ConstantOfShape."""
        node.op_type = "Full"
        return self.visit(node, input_vars, **kwargs)

    def visit_Transpose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Transpose."""
        fmt = self._format_transpose(kwargs)  # pragma: no cover
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)  # pragma: no cover

    def visit_RandomCategorical(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle RandomCategorical."""
        fmt = self._format_random_categorical(input_vars, kwargs)  # pragma: no cover
        return OpFormatter.format_backend_string(fmt, input_vars, kwargs)  # pragma: no cover

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "mx"

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        ops_map = self._SIMPLE_OPS_MAP.copy()
        ops_map["Transpose"] = self._format_transpose(kwargs)

        ops_map["Beta"] = "mx.random.uniform(shape={shape})"
        ops_map["Dirichlet"] = "mx.random.uniform(shape={shape})"
        ops_map["Gamma"] = "mx.random.uniform(shape={shape})"
        ops_map["RngBitGenerator"] = "mx.random.randint(0, 255, {shape})"
        ops_map["RngUniform"] = "mx.random.uniform(low={0}, high={1}, shape={shape})"

        ops_map["Infeed"] = "{0}"
        ops_map["Outfeed"] = "{0}"
        ops_map["AxisIndex"] = "0"
        ops_map["WithShardingConstraint"] = "{0}"
        return ops_map

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = mx.array({val_repr})")

    _forward_method_name = "__call__"

    def _get_prefix_code(self) -> list[str]:
        """Function docstring."""
        from ml_switcheroo_compiler.backends.common.mixins.nn import NNASTVisitor

        res = [
            "import mlx.core as mx",
            "import mlx.nn as nn\n",
            *_MLX_RESIZE_TMPL.split("\n"),
            *_MLX_ISTFT_TMPL.split("\n"),
            *_MLX_MEL_FILTERBANK_TMPL.split("\n"),
            *_MLX_MFCC_TMPL.split("\n"),
            *_MLX_POWER_ITERATION_TMPL.split("\n"),
        ]
        res.extend(
            NNASTVisitor(generator=self)._get_group_norm_code(
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
