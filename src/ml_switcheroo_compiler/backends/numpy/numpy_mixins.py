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
from ml_switcheroo_compiler.ir.core import IRNode


class NumpyVisionMixin:
    """Mixin."""

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"np_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"np_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"  # pragma: no cover

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"np_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"  # pragma: no cover

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(
            node
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"np_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"  # pragma: no cover

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (  # pragma: no cover
            _extract_extract_boxes_attributes(node)  # pragma: no cover
        )  # pragma: no cover
        df_str = "None" if data_format is None else f'"{data_format}"'  # pragma: no cover
        return f"np_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"  # pragma: no cover

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")  # pragma: no cover
        return (
            f"np_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"  # pragma: no cover
        )

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")  # pragma: no cover
        iou_threshold = node.attributes.get("iou_threshold", 0.5)  # pragma: no cover
        score_threshold = node.attributes.get("score_threshold", float("-inf"))  # pragma: no cover
        return f"np_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"  # pragma: no cover

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        return f"np_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"  # pragma: no cover

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")  # pragma: no cover
        align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
        return (
            f"np_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"  # pragma: no cover
        )


class NumpyAudioMixin:
    """Mixin."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(
            node
        )  # pragma: no cover
        return f"np_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"  # pragma: no cover

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
        return f"np_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"  # pragma: no cover

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
        return f"np_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"  # pragma: no cover


class NumpyScatterMixin:
    """Mixin."""

    def visit_TensorScatterUpdate(
        self, node: IRNode, input_vars: list[str], **kwargs: object
    ) -> str:
        """Handle TensorScatterUpdate."""
        return f"(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"  # pragma: no cover

    def visit_TensorScatterAdd(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterAdd."""
        return f"(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"  # pragma: no cover

    def visit_TensorScatterMax(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterMax."""
        return f"(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"  # pragma: no cover

    def visit_TensorScatterMin(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterMin."""
        return f"(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"  # pragma: no cover
