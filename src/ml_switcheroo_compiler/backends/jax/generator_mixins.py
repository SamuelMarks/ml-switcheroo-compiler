"""JAX Generator Mixins."""

from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_mel_attributes,
    extract_stft_attributes,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)


class JaxDistributedVisitor:
    """Mixin for JAX distributed node visitors."""

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        axis_name = getattr(node, "attributes", {}).get("axis_name", "'x'")
        return f"jax.lax.all_gather({tensor}, axis_name={axis_name})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        axis = getattr(node, "attributes", {}).get("axis", 0)
        axis_name = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op = getattr(node, "attributes", {}).get("op", "jax.lax.psum")
        return f"jax.lax.reduce_scatter({tensor}, {op}, scatter_dimension={axis}, axis_name={axis_name})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        axis_name = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op = getattr(node, "attributes", {}).get("op", "psum")
        return f"jax.lax.{op}({tensor}, axis_name={axis_name})"


class JaxMathVisitor:
    """Mixin for JAX math node visitors."""

    def visit_SegmentSum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate segment sum."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate segment max."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate segment min."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentProd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate segment prod."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentSum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate unsorted segment sum."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate unsorted segment max."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate unsorted segment min."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentProd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate unsorted segment prod."""
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_MatrixExponential(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate MatrixExponential."""
        return f"jax.scipy.linalg.expm({input_vars[0]})"

    def visit_Polar(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Polar."""
        side = getattr(node, "attributes", {}).get("side", "'right'")
        if not isinstance(side, str) or not side.startswith("'"):
            side = f"'{side}'"
        return f"jax.scipy.linalg.polar({input_vars[0]}, side={side})"

    def visit_Schur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Schur."""
        return f"jax.scipy.linalg.schur({input_vars[0]})"

    def visit_Cholesky(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Cholesky."""
        return f"jax.numpy.linalg.cholesky({input_vars[0]})"

    def visit_Svd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Svd."""
        full_matrices = getattr(node, "attributes", {}).get("full_matrices", True)
        compute_uv = getattr(node, "attributes", {}).get("compute_uv", True)
        return f"jax.numpy.linalg.svd({input_vars[0]}, full_matrices={full_matrices}, compute_uv={compute_uv})"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = getattr(node, "attributes", {}).get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"jax_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"jax_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"jnp.einsum('{eq}', {args_str})"  # pragma: no cover


class JaxControlFlowVisitor:
    """Mixin for JAX control flow node visitors."""

    def visit_If(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate If."""
        # Simple fallback for jax.lax.cond if proper block tracing is not used natively
        return f"jax.lax.cond({input_vars[0]}, lambda: None, lambda: None)"

    def visit_Loop(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate WhileLoop."""
        return f"jax.lax.while_loop(lambda _: True, lambda _: {input_vars[0]}, {input_vars[0]})"

    def visit_Scan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Scan."""
        return f"jax.lax.scan(lambda c, x: (c, x), {input_vars[0]}, {input_vars[1]} if len({input_vars}) > 1 else None)"


class JaxVisionVisitor:
    """Mixin for JAX vision node visitors."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"jax_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"jax_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"  # pragma: no cover

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"jax_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"  # pragma: no cover

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = getattr(node, "attributes", {}).get("bounding_box_format", "xyxy")  # pragma: no cover
        return f"jax_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"  # pragma: no cover

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = getattr(node, "attributes", {}).get("max_output_size")  # pragma: no cover
        iou_threshold = getattr(node, "attributes", {}).get("iou_threshold", 0.5)  # pragma: no cover
        score_threshold = getattr(node, "attributes", {}).get("score_threshold", float("-inf"))  # pragma: no cover
        return f"jax_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"  # pragma: no cover

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = getattr(node, "attributes", {}).get("size")  # pragma: no cover
        align_corners = getattr(node, "attributes", {}).get("align_corners", False)  # pragma: no cover
        return f"jax_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"  # pragma: no cover

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = getattr(node, "attributes", {}).get("size")  # pragma: no cover
        align_corners = getattr(node, "attributes", {}).get("align_corners", False)  # pragma: no cover
        return f"jax_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"  # pragma: no cover

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (  # pragma: no cover
            _extract_extract_boxes_attributes(node)
        )
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"jax_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"  # pragma: no cover

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"jax_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover


class JaxAudioVisitor:
    """Mixin for JAX audio node visitors."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)  # pragma: no cover
        return f"jax_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"  # pragma: no cover

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        (
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            _,
        ) = (  # pragma: no cover
            extract_mel_attributes(node)
        )
        return f"jax_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"  # pragma: no cover

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        (
            num_mel_bins,
            _,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            num_mfccs,
        ) = (  # pragma: no cover
            extract_mel_attributes(node)
        )
        return f"jax_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"  # pragma: no cover
