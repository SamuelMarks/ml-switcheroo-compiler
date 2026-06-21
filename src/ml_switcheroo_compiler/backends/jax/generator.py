from ml_switcheroo_compiler.backends.common.generator_mixins import GroupNormConfig

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

"""JAX/Flax Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("jax")
class JAXCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """JAX code generator."""

    def _get_backend_prefix(self) -> str:
        return "jax"

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        axis_name = node.attributes.get("axis_name", "'x'")
        return f"jax.lax.all_gather({tensor}, axis_name={axis_name})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        axis = node.attributes.get("axis", 0)
        axis_name = node.attributes.get("axis_name", "'x'")
        op = node.attributes.get("op", "jax.lax.psum")
        return f"jax.lax.reduce_scatter({tensor}, {op}, scatter_dimension={axis}, axis_name={axis_name})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        axis_name = node.attributes.get("axis_name", "'x'")
        op = node.attributes.get("op", "psum")
        return f"jax.lax.{op}({tensor}, axis_name={axis_name})"

    """Emit JAX-compatible pure functions from IR."""

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        res = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        res = "jnp.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"jax_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (
            _extract_extract_boxes_attributes(node)
        )
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

    def visit_IoU(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate iou."""
        bounding_box_format = node.attributes.get("bounding_box_format", "xyxy")
        return f"jax_iou({input_vars[0]}, {input_vars[1]}, '{bounding_box_format}')"

    def visit_NonMaxSuppression(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate nms."""
        max_output_size = node.attributes.get("max_output_size")
        iou_threshold = node.attributes.get("iou_threshold", 0.5)
        score_threshold = node.attributes.get("score_threshold", float("-inf"))
        return f"jax_nms({input_vars[0]}, {input_vars[1]}, {max_output_size}, {iou_threshold}, {score_threshold})"

    def visit_ResizeBicubic(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize bicubic."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'bicubic', {align_corners})"

    def visit_ResizeLanczos3(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate resize lanczos3."""
        size = node.attributes.get("size")
        align_corners = node.attributes.get("align_corners", False)
        return f"jax_resize({input_vars[0]}, {size}, 'lanczos3', {align_corners})"

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"jax_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, _ = (
            extract_mel_attributes(node)
        )
        return f"jax_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        num_mel_bins, _, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs = (
            extract_mel_attributes(node)
        )
        return f"jax_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"jax_perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"jnp.einsum('{eq}', {args_str})"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated JAX Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "jnp.matmul({0}, {1})",
            "Dot": "jnp.dot({0}, {1})",
            "BroadcastTo": "jnp.broadcast_to({0}, {shape})",
            "Reshape": "jnp.reshape({0}, {shape})",
            "TrueDivide": "jnp.true_divide({0}, {1})",
            "Arange": "jnp.arange({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "jnp.sort({0}, axis={dimension})",
            "ArgSort": "jnp.argsort({0}, axis={dimension})",
            "Allclose": "jnp.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "jnp.fft.fft({0})",
            "Rfft": "jnp.fft.rfft({0})",
            "Fftn": "jnp.fft.fftn({0})",
            "Erfinv": "jax.scipy.special.erfinv({0})",
            "NanToNum": "jnp.nan_to_num({0}, nan={nan}, posinf={posinf}, neginf={neginf})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "TensorScatterUpdate": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].set({2})",
            "TensorScatterAdd": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].add({2})",
            "TensorScatterMax": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].max({2})",
            "TensorScatterMin": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].min({2})",
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(prefix="jnp", op_type=op_type, input_vars=input_vars, kwargs=kwargs)
        )

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Generate file header with module docstrings."""
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports."""
        from ml_switcheroo_compiler.backends.jax.jax_prefix_template import JAX_PREFIX_TEMPLATE

        return [
            "import jax",
            "import jax.numpy as jnp",
            *self._get_group_norm_code(
                GroupNormConfig(
                    "jax",
                    "jax.numpy as jnp",
                    "jnp.reshape",
                    "jnp.mean",
                    "jnp.var",
                    "jnp.sqrt",
                    "axis",
                    "keepdims",
                )
            ),
            "import jax.scipy.special",
            *JAX_PREFIX_TEMPLATE.split("\n"),
        ]

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs):")
        self.indent_level += 1

    def _traverse_ir_graph(self) -> None:
        """Core iteration loop that traverses the IR graph."""
        # The actual traversal and return generation is handled by visitor's _generate_body
        self._generate_body()

    def _generate_return_block(self) -> None:
        """Format the final return statement (delegated to visitor)."""
        pass  # Delegate to _generate_body

    def generate(self) -> str:
        """Generate JAX code from the IR graph.

        Returns:
            str: The generated JAX Python code
        """
        self.code = self._generate_file_header() + self._resolve_imports()
        self._generate_function_signature()
        self._traverse_ir_graph()
        self._generate_return_block()
        return "\n".join(self.code)
