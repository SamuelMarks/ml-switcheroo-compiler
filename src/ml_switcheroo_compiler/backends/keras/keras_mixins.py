"""Mixins."""

from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)
from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_stft_attributes,
    extract_mel_attributes,
)


class KerasVisionMixin:
    """Mixin."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"keras_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return (  # pragma: no cover
            f"keras_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"
        )

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"keras_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"  # pragma: no cover

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (  # pragma: no cover
            _extract_extract_boxes_attributes(node)  # pragma: no cover
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"keras_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"  # pragma: no cover

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")  # pragma: no cover
        return f"keras_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"  # pragma: no cover

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")  # pragma: no cover
        iou_threshold = node.attributes.get("iou_threshold", 0.5)  # pragma: no cover
        score_threshold = node.attributes.get("score_threshold", float("-inf"))  # pragma: no cover
        return f"keras_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"  # pragma: no cover

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        return (
            f"keras_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"  # pragma: no cover
        )

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        return f"keras_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"  # pragma: no cover

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        """Generate keras.ops.image.perspective_transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"keras.ops.image.perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, interpolation='{interpolation}', fill_value={fill_value}, data_format={df_str})"  # pragma: no cover


class KerasAudioMixin:
    """Mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(
            node
        )  # pragma: no cover
        return f"keras_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"  # pragma: no cover

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
            extract_mel_attributes(node)  # pragma: no cover
        )  # pragma: no cover
        return f"keras_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"  # pragma: no cover

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
            extract_mel_attributes(node)  # pragma: no cover
        )  # pragma: no cover
        return f"keras_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"  # pragma: no cover
