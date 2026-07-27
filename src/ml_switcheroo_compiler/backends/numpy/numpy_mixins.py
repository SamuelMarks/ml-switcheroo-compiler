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
from ml_switcheroo_compiler.ir.core import IRNode


class NumpyVisionVisitor:
    """Provides AST visitor methods for vision operations in NumPy."""

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a perspective transform operation.

        Args:
            node: The IR node representing the perspective transform.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, PerspectiveConfig(interpolation='{interpolation}', fill_value={fill_value}, data_format={df_str}))"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for an elastic transform operation.

        Args:
            node: The IR node representing the elastic transform.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a gaussian blur operation.

        Args:
            node: The IR node representing the gaussian blur.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a median filter operation.

        Args:
            node: The IR node representing the median filter.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for extracting bounding boxes from images.

        Args:
            node: The IR node representing the extract bounding boxes operation.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        crop_size, interpolation, extrapolation_value, data_format = _extract_extract_boxes_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"np_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for calculating Intersection over Union (IoU).

        Args:
            node: The IR node representing the IoU operation.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"np_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for non-maximum suppression.

        Args:
            node: The IR node representing the non-maximum suppression.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"np_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a bicubic resize operation.

        Args:
            node: The IR node representing the bicubic resize.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"np_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a Lanczos3 resize operation.

        Args:
            node: The IR node representing the Lanczos3 resize.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"np_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"


class NumpyAudioVisitor:
    """Provides AST visitor methods for audio operations in NumPy."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for the inverse short-time Fourier transform.

        Args:
            node: The IR node representing the ISTFT.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"np_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a mel filterbank.

        Args:
            node: The IR node representing the mel filterbank.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        (
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            _,
        ) = extract_mel_attributes(node)
        return f"np_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for calculating Mel-frequency cepstral coefficients.

        Args:
            node: The IR node representing the MFCC operation.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        (
            num_mel_bins,
            _,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            num_mfccs,
        ) = extract_mel_attributes(node)
        return f"np_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"


class NumpyScatterVisitor:
    """Provides AST visitor methods for scatter operations in NumPy."""

    def visit_TensorScatterUpdate(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a tensor scatter update operation.

        Args:
            node: The IR node representing the tensor scatter update.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a tensor scatter add operation.

        Args:
            node: The IR node representing the tensor scatter add.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMax(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a tensor scatter max operation.

        Args:
            node: The IR node representing the tensor scatter max.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generates NumPy code for a tensor scatter min operation.

        Args:
            node: The IR node representing the tensor scatter min.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"
