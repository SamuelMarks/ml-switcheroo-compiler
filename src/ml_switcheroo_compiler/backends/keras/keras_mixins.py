# ruff: noqa: E501
"""Mixins."""

from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_mel_attributes,
    extract_stft_attributes,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)


class KerasVisionVisitor:
    """Mixin."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = _extract_extract_boxes_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, BoundingBoxExtractionConfig(crop_size={crop_size}, interpolation='{interpolation}', extrapolation_value={extrapolation_value}, data_format={df_str}))"

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

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate perspective transform."""
        """Generate keras.ops.image.perspective_transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras.ops.image.perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, interpolation='{interpolation}', fill_value={fill_value}, data_format={df_str})"


class KerasAudioVisitor:
    """Mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"keras_istft({input_vars[0]}, STFTConfig(frame_length={frame_length}, frame_step={frame_step}, fft_length={fft_len_str}, window='{window}', center={center}))"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        (
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            _,
        ) = extract_mel_attributes(node)
        return f"keras_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        (
            num_mel_bins,
            _,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            num_mfccs,
        ) = extract_mel_attributes(node)
        return f"keras_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"
