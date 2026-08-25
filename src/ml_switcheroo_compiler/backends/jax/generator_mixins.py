# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
    """Provide mixin for JAX distributed node visitors."""

    def visit_Send(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Send tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: JAX code for send via host callback.
        """
        dst: object = node.attributes.get("dst_rank", 0)
        self.code.append(f"    # JAX Send to {dst} (via host_callback or explicit MPI bridge)")
        return ""

    def visit_Recv(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Receive tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: JAX code for recv via host callback.
        """
        src: object = node.attributes.get("src_rank", 0)
        shape: object = node.attributes.get("shape", ())
        dtype: object = "jnp." + str(node.attributes.get("dtype", "float32")).lower()
        nid: object = getattr(node, "id", "")
        res_var: object = f"v_{nid.replace('-', '_')}"
        self.code.append(f"    {res_var} = jnp.zeros({list(shape)}, dtype={dtype}) # JAX Recv from {src} placeholder")
        return res_var

    """Provide mixin for JAX distributed node visitors."""

    def visit_AllGather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the all_gather operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        tensor: object = input_vars[0]
        axis_name: object = getattr(node, "attributes", {}).get("axis_name", "'x'")
        return f"jax.lax.all_gather({tensor}, axis_name={axis_name})"

    def visit_ReduceScatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the reduce_scatter operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        tensor: object = input_vars[0]
        axis: object = getattr(node, "attributes", {}).get("axis", 0)
        axis_name: object = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op: object = getattr(node, "attributes", {}).get("op", "jax.lax.psum")
        return f"jax.lax.reduce_scatter({tensor}, {op}, scatter_dimension={axis}, axis_name={axis_name})"

    def visit_AllReduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the all_reduce operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        tensor: object = input_vars[0]
        axis_name: object = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op: object = getattr(node, "attributes", {}).get("op", "psum")
        return f"jax.lax.{op}({tensor}, axis_name={axis_name})"


class JaxMathVisitor:
    """Provide mixin for JAX math node visitors."""

    def visit_SegmentSum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the SegmentSum operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the SegmentMax operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the SegmentMin operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentProd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the SegmentProd operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentSum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the UnsortedSegmentSum operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the UnsortedSegmentMax operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the UnsortedSegmentMin operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentProd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the UnsortedSegmentProd operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_segments: object = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_MatrixExponential(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the MatrixExponential operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax.scipy.linalg.expm({input_vars[0]})"

    def visit_Polar(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Polar operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        side: object = getattr(node, "attributes", {}).get("side", "'right'")
        if not isinstance(side, str) or not side.startswith("'"):
            side: object = f"'{side}'"
        return f"jax.scipy.linalg.polar({input_vars[0]}, side={side})"

    def visit_Schur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Schur operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax.scipy.linalg.schur({input_vars[0]})"

    def visit_Cholesky(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Cholesky operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax.numpy.linalg.cholesky({input_vars[0]})"

    def visit_Svd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Svd operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        full_matrices: object = getattr(node, "attributes", {}).get("full_matrices", True)
        compute_uv: object = getattr(node, "attributes", {}).get("compute_uv", True)
        return f"jax.numpy.linalg.svd({input_vars[0]}, full_matrices={full_matrices}, compute_uv={compute_uv})"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the PowerIteration operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        num_iters: object = getattr(node, "attributes", {}).get("num_iters", 1)
        u_var: object = input_vars[1] if len(input_vars) > 1 else "None"
        return f"jax_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the RaggedDot operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Einsum operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        args_str: object = ", ".join(input_vars)
        eq: object = kwargs.get("equation", "")
        return f"jnp.einsum('{eq}', {args_str})"


class JaxControlFlowVisitor:
    """Provide mixin for JAX control flow node visitors."""

    def visit_If(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the If operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        # Simple fallback for jax.lax.cond if proper block tracing is not used natively
        return f"jax.lax.cond({input_vars[0]}, lambda: None, lambda: None)"

    def visit_Loop(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Loop operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax.lax.while_loop(lambda _: True, lambda _: {input_vars[0]}, {input_vars[0]})"

    def visit_Scan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Scan operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        return f"jax.lax.scan(lambda c, x: (c, x), {input_vars[0]}, {input_vars[1] if len(input_vars) > 1 else None})"


class JaxVisionVisitor:
    """Provide mixin for JAX vision node visitors."""

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the ElasticTransform operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str: object = "None" if data_format is None else f'"{data_format}"'
        return f"jax_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the GaussianBlur operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str: object = "None" if data_format is None else f'"{data_format}"'
        return f"jax_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the MedianFilter operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str: object = "None" if data_format is None else f'"{data_format}"'
        return f"jax_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the IoU operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        bounding_box_format: object = getattr(node, "attributes", {}).get("bounding_box_format", "xyxy")
        return f"jax_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the NonMaxSuppression operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        max_output_size: object = getattr(node, "attributes", {}).get("max_output_size")
        iou_threshold: object = getattr(node, "attributes", {}).get("iou_threshold", 0.5)
        score_threshold: object = getattr(node, "attributes", {}).get("score_threshold", float("-inf"))
        return f"jax_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the ResizeBicubic operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        size: object = getattr(node, "attributes", {}).get("size")
        align_corners: object = getattr(node, "attributes", {}).get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the ResizeLanczos3 operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        size: object = getattr(node, "attributes", {}).get("size")
        align_corners: object = getattr(node, "attributes", {}).get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_ExtractBoundingBoxes(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the ExtractBoundingBoxes operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        crop_size, interpolation, extrapolation_value, data_format = _extract_extract_boxes_attributes(node)
        df_str: object = "None" if data_format is None else f'"{data_format}"'
        return f"jax_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_PerspectiveTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the PerspectiveTransform operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str: object = "None" if data_format is None else f'"{data_format}"'
        return f"jax_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"


class JaxAudioVisitor:
    """Provide mixin for JAX audio node visitors."""

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Istft operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"jax_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the MelFilterbank operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        (
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            _,
        ) = extract_mel_attributes(node)
        return f"jax_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate JAX code for the Mfcc operation.

        Args:
            node: The intermediate representation node representing the operation.
            input_vars: A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            A string containing the generated JAX code.
        """
        (
            num_mel_bins,
            _,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            num_mfccs,
        ) = extract_mel_attributes(node)
        return f"jax_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"
