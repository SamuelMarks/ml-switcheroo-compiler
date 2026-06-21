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

"""Keras Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.formatters import OpFormatter
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


@register_backend("keras")
class KerasCodeGenerator(SharedASTGeneratorMixin, BaseGenerator):
    """Emit Keras Functional API script from IR."""

    def _get_backend_prefix(self) -> str:
        return "keras"

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes the object.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        res = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        res = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs: object) -> str:
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"
        return "keras.ops.transpose({0})"

    def visit_ElasticTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate elastic transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_elastic_transform({input_vars[0]}, {input_vars[1]}, '{interpolation}', {fill_value}, {df_str})"

    def visit_GaussianBlur(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate gaussian blur."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return (
            f"keras_gaussian_blur({input_vars[0]}, {kernel_size}, {sigma}, '{padding}', {df_str})"
        )

    def visit_MedianFilter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate median filter."""
        kernel_size, sigma, padding, data_format = _extract_filter_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_median_filter({input_vars[0]}, {kernel_size}, '{padding}', {df_str})"

    def visit_ExtractBoundingBoxes(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate extract bounding boxes."""
        crop_size, interpolation, extrapolation_value, data_format = (
            _extract_extract_boxes_attributes(node)
        )
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras_extract_bounding_boxes({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {crop_size}, '{interpolation}', {extrapolation_value}, {df_str})"

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

    def visit_Istft(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate istft."""
        frame_length, frame_step, _, window, center, fft_len_str = extract_stft_attributes(node)
        return f"keras_istft({input_vars[0]}, {frame_length}, {frame_step}, {fft_len_str}, '{window}', {center})"

    def visit_MelFilterbank(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mel_filterbank."""
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, _ = (
            extract_mel_attributes(node)
        )
        return f"keras_mel_filterbank({num_mel_bins}, {num_spectrogram_bins}, {sample_rate}, {lower_edge_hertz}, {upper_edge_hertz})"

    def visit_Mfcc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate mfcc."""
        num_mel_bins, _, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs = (
            extract_mel_attributes(node)
        )
        return f"keras_mfcc({input_vars[0]}, {sample_rate}, {num_mel_bins}, {lower_edge_hertz}, {upper_edge_hertz}, {num_mfccs})"

    def visit_PerspectiveTransform(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate perspective transform."""
        """Generate keras.ops.image.perspective_transform."""
        interpolation, fill_value, data_format = _extract_vision_transform_attributes(node)
        df_str = "None" if data_format is None else f'"{data_format}"'
        return f"keras.ops.image.perspective_transform({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, interpolation='{interpolation}', fill_value={fill_value}, data_format={df_str})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"keras.ops.einsum('{eq}', {args_str})"

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Fallback for generic nodes.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated Keras Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = {
            "Matmul": "keras.ops.matmul({0}, {1})",
            "Dot": "keras.ops.dot({0}, {1})",
            "BroadcastTo": "keras.ops.broadcast_to({0}, {shape})",
            "Reshape": "keras.ops.reshape({0}, {shape})",
            "TrueDivide": "keras.ops.true_divide({0}, {1})",
            "Arange": "keras.ops.arange({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "keras.ops.sort({0}, axis={dimension})",
            "ArgSort": "keras.ops.argsort({0}, axis={dimension})",
            "Allclose": "keras.ops.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fft": "keras.ops.fft.fft({0})",
            "Rfft": "keras.ops.fft.rfft({0})",
            "Fftn": "keras.ops.fft.fftn({0})",
            "Erfinv": "keras.ops.erfinv({0})",
            "NanToNum": "keras.ops.where(keras.ops.isnan({0}), {nan}, keras.ops.where(keras.ops.isinf({0}) & ({0} > 0), {posinf}, keras.ops.where(keras.ops.isinf({0}) & ({0} < 0), {neginf}, {0})))",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": self._format_transpose(kwargs),
        }

        if op_type in ops_map:
            fmt = ops_map[op_type]
            return OpFormatter.format_backend_string(fmt, input_vars, kwargs)

        from ml_switcheroo_compiler.backends.formatters import FormatterContext

        return OpFormatter.format_generic_fallback(
            FormatterContext(
                prefix="keras.ops", op_type=op_type, input_vars=input_vars, kwargs=kwargs
            )
        )

    def _emit_input_assignment(
        self,
        var_name: str,
        node: IRNode,
        input_prefix: str,
        input_idx: int,
    ) -> None:
        """Evaluate emit input assignment.

        Args:
            var_name (str): Argument var_name
            node (IRNode): Argument node
            input_prefix (str): Argument input_prefix
            input_idx (int): Argument input_idx
        """
        shape_str = (
            str(node.shape_metadata)
            if hasattr(node, "shape_metadata") and node.shape_metadata
            else "(None,)"
        )
        self.add_line(f"{var_name} = keras.Input(shape={shape_str}, name='{node.id}')")
        self.keras_input_vars.append(var_name)

    def _emit_body_return(self, returns: list[str]) -> None:
        pass

    def _emit_output_assignment(
        self,
        node: IRNode,
        input_vars: list[str],
        returns: str,
    ) -> None:
        """Evaluate emit output assignment.

        Args:
            node (IRNode): Argument node
            input_vars (list[str]): Argument input_vars
            returns (str): Argument returns
        """
        self.keras_output_vars.extend(input_vars)

    def _generate_file_header(self) -> list[str]:
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        from ml_switcheroo_compiler.backends.keras.keras_prefix_template import (
            KERAS_PREFIX_TEMPLATE,
        )

        return [
            "import keras\n",
            *self._get_group_norm_code(
                GroupNormConfig(
                    "keras",
                    "keras.ops",
                    "keras.ops.reshape",
                    "keras.ops.mean",
                    "keras.ops.var",
                    "keras.ops.sqrt",
                    "axis",
                    "keepdims",
                )
            ),
            *KERAS_PREFIX_TEMPLATE.split("\n"),
        ]

    def _generate_function_signature(self) -> None:
        self.indent_level = 0
        self.add_line("def get_model():")
        self.keras_input_vars = []
        self.keras_output_vars = []
        self.indent_level += 1

    def _traverse_ir_graph(self) -> None:
        self._generate_body()

    def _generate_return_block(self) -> None:
        inputs_str = ", ".join(self.keras_input_vars)
        outputs_str = ", ".join(self.keras_output_vars)
        self.add_line(
            f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])",
        )

    def generate(self) -> str:
        """Generate Keras model code from the IR graph.

        Returns:
            str: The generated Keras Python code
        """
        self.code = self._generate_file_header() + self._resolve_imports()
        self._generate_function_signature()
        self._traverse_ir_graph()
        self._generate_return_block()
        return "\n".join(self.code)
