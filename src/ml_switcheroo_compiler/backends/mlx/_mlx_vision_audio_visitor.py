"""Vision and audio operation visitor mixins for the MLX code generator."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)


class MLXVisionVisitor:
    """MLX Vision ops visitor mixin."""

    def visit_ElasticTransform(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the ElasticTransform operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ElasticTransform operation.
        """
        (interpolation, fill_value, data_format) = _extract_vision_transform_attributes(node)
        df_str: str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the GaussianBlur operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the GaussianBlur operation.
        """
        (kernel_size, sigma, padding, data_format) = _extract_filter_attributes(node)
        df_str: str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the MedianFilter operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the MedianFilter operation.
        """
        (kernel_size, sigma, padding, data_format) = _extract_filter_attributes(node)
        df_str: str = "None" if data_format is None else f'"{data_format}"'
        return f"mlx_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the ExtractBoundingBoxes operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ExtractBoundingBoxes operation.
        """
        (crop_size, interpolation, extrapolation_value, data_format) = _extract_extract_boxes_attributes(node)
        df_str: str = "None" if data_format is None else f'"{data_format}"'
        return (
            f"mlx_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, "
            f"{crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"
        )

    def visit_IoU(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the IoU operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the IoU operation.
        """
        bounding_box_format: str = coerce_str(node.attributes.get("bounding_box_format", "xyxy"))
        return f"mlx_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the NonMaxSuppression operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the NonMaxSuppression operation.
        """
        raw_max_out = node.attributes.get("max_output_size")
        max_output_size: int | None = None if raw_max_out is None else coerce_int(raw_max_out)
        iou_threshold: float = coerce_float(node.attributes.get("iou_threshold", 0.5))
        score_threshold: float = coerce_float(node.attributes.get("score_threshold", float("-inf")))
        return f"mlx_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the ResizeBicubic operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ResizeBicubic operation.
        """
        raw_size = node.attributes.get("size")
        size: int | tuple[int, ...] | list[int] | None = (
            None
            if raw_size is None
            else (
                [coerce_int(x) for x in raw_size]
                if isinstance(raw_size, list)
                else (tuple(coerce_int(x) for x in raw_size) if isinstance(raw_size, tuple) else coerce_int(raw_size))
            )
        )
        align_corners: bool = coerce_bool(node.attributes.get("align_corners", False))
        return f"mlx_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the ResizeLanczos3 operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ResizeLanczos3 operation.
        """
        raw_size = node.attributes.get("size")
        size: int | tuple[int, ...] | list[int] | None = (
            None
            if raw_size is None
            else (
                [coerce_int(x) for x in raw_size]
                if isinstance(raw_size, list)
                else (tuple(coerce_int(x) for x in raw_size) if isinstance(raw_size, tuple) else coerce_int(raw_size))
            )
        )
        align_corners: bool = coerce_bool(node.attributes.get("align_corners", False))
        return f"mlx_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_PerspectiveTransform(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the PerspectiveTransform operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the PerspectiveTransform operation.
        """
        (interpolation, fill_value, data_format) = _extract_vision_transform_attributes(node)
        df_str: str = "None" if data_format is None else f'"{data_format}"'
        return (
            f"mlx_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, "
            f"'{interpolation}', {fill_value}, {df_str})"
        )


class MLXAudioVisitor:
    """MLX Audio ops visitor mixin."""

    def visit_Istft(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Istft operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Istft operation.
        """
        from ml_switcheroo_compiler.backends.common.audio_utils import extract_stft_attributes

        (frame_length, frame_step, _, window, center, fft_len_str) = extract_stft_attributes(node)
        return (
            f"mlx_istft({input_vars[0]}, STFTConfig({frame_length}, {frame_step}, {fft_len_str}, '{window}', {center}))"
        )

    def visit_MelFilterbank(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the MelFilterbank operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the MelFilterbank operation.
        """
        num_mel_bins: int = coerce_int(node.attributes.get("num_mel_bins"))
        num_spectrogram_bins: int = coerce_int(node.attributes.get("num_spectrogram_bins"))
        sample_rate: int = coerce_int(node.attributes.get("sample_rate"))
        raw_low = node.attributes.get("lower_edge_hertz")
        lower_edge_hertz: int | float = (
            raw_low if isinstance(raw_low, int) and not isinstance(raw_low, bool) else coerce_float(raw_low)
        )
        raw_high = node.attributes.get("upper_edge_hertz")
        upper_edge_hertz: int | float = (
            raw_high if isinstance(raw_high, int) and not isinstance(raw_high, bool) else coerce_float(raw_high)
        )
        return (
            f"mlx_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, "
            f"{lower_edge_hertz}, {upper_edge_hertz})"
        )

    def visit_Mfcc(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Mfcc operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Mfcc operation.
        """
        raw_sr = node.attributes.get("sample_rate")
        sample_rate: int | None = None if raw_sr is None else coerce_int(raw_sr)
        num_mel_bins: int = coerce_int(node.attributes.get("num_mel_bins", 40))
        raw_low = node.attributes.get("lower_edge_hertz", 20.0)
        lower_edge_hertz: int | float = (
            raw_low if isinstance(raw_low, int) and not isinstance(raw_low, bool) else coerce_float(raw_low)
        )
        raw_high = node.attributes.get("upper_edge_hertz", 4000.0)
        upper_edge_hertz: int | float = (
            raw_high if isinstance(raw_high, int) and not isinstance(raw_high, bool) else coerce_float(raw_high)
        )
        num_mfccs: int = coerce_int(node.attributes.get("num_mfccs", 13))
        return (
            f"mlx_mfcc({input_vars[0]}, MFCCConfig({sample_rate}, {num_mel_bins}, "
            f"{lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs}))"
        )
