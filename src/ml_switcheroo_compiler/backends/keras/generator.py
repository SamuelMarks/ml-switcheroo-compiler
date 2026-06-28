# ruff: noqa: E402

"""Module docstring."""

from ml_switcheroo_compiler.backends.common.generator_mixins import GroupNormConfig

"""Keras Target Emission."""
from .keras_mixins import KerasVisionMixin, KerasAudioMixin
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode


class KerasSignatureBuilder:
    """Helper for building Keras Model signatures."""

    @staticmethod
    def get_input_assignment(var_name: str, node: IRNode) -> str:
        """Function docstring."""
        shape_str = (
            str(node.shape_metadata)
            if hasattr(node, "shape_metadata") and node.shape_metadata
            else "(None,)"
        )
        return f"{var_name} = keras.Input(shape={shape_str}, name='{node.id}')"

    @staticmethod
    def get_return_block(input_vars: list[str], output_vars: list[str]) -> str:
        """Function docstring."""
        inputs_str = ", ".join(input_vars)
        outputs_str = ", ".join(output_vars)
        return f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])"


class KerasTensorManipulator:
    """Helper for tensor manipulations."""

    @staticmethod
    def format_zeros_like(op: str, kwargs: object) -> str:
        """Function docstring."""
        res = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    @staticmethod
    def format_full(kwargs: object) -> str:
        """Function docstring."""
        res = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    @staticmethod
    def format_transpose(kwargs: object) -> str:
        """Function docstring."""
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"  # pragma: no cover
        return "keras.ops.transpose({0})"


@register_backend("keras")
class KerasCodeGenerator(SharedASTGeneratorMixin, BaseGenerator, KerasVisionMixin, KerasAudioMixin):
    """Emit Keras Functional API script from IR."""

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "keras"  # pragma: no cover

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes the object.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def visit_ConvTranspose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ConvTranspose."""
        lhs = input_vars[0]
        rhs = input_vars[1]
        strides = node.attributes.get("strides", 1)
        padding = node.attributes.get("padding", "VALID")
        return f"keras_conv_transpose({lhs}, {rhs}, {strides}, '{padding}')"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"keras_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"keras.ops.einsum('{eq}', {args_str})"  # pragma: no cover

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "keras.ops"

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        return {
            "Infeed": "{0}",
            "Outfeed": "{0}",
            "AxisIndex": "0",
            "WithShardingConstraint": "{0}",
            "Beta": "keras.random.beta({shape}, {1}, {2})",
            "Dirichlet": "keras.random.dirichlet({shape}, {1})",
            "Gamma": "keras.random.gamma({shape}, {1})",
            "RngBitGenerator": "keras.random.randint({shape}, 0, 255)",
            "RngUniform": "keras.random.uniform({shape}, minval={0}, maxval={1})",
            "Matmul": "keras.ops.matmul({0}, {1})",
            "Trace": "tf.linalg.trace",
            "Adjoint": "tf.linalg.adjoint",
            "BandPart": "tf.linalg.band_part",
            "CholeskySolve": "tf.linalg.cholesky_solve",
            "TriInv": "tf.linalg.inv({0})",
            "BandedTriangularSolve": "tf.linalg.banded_triangular_solve",
            "EighTridiagonal": "tf.linalg.eigh_tridiagonal",
            "MatrixRank": "tf.linalg.matrix_rank",
            "MatrixTranspose": "tf.linalg.matrix_transpose",
            "Sqrtm": "tf.linalg.sqrtm",
            "Dot": "keras.ops.dot({0}, {1})",
            "BroadcastTo": "keras.ops.broadcast_to({0}, {shape})",
            "Reshape": "keras.ops.reshape({0}, {shape})",
            "TruncateDiv": "keras.ops.trunc(keras.ops.divide({0}, {1}))",
            "TruncateMod": "keras.ops.mod({0}, {1})",
            "TrueDivide": "keras.ops.true_divide({0}, {1})",
            "Arange": "keras.ops.arange({0})",
            "Zeros": KerasTensorManipulator.format_zeros_like("zeros", kwargs),
            "Ones": KerasTensorManipulator.format_zeros_like("ones", kwargs),
            "Full": KerasTensorManipulator.format_full(kwargs),
            "Sort": "keras.ops.sort({0}, axis={dimension})",
            "ArgSort": "keras.ops.argsort({0}, axis={dimension})",
            "Allclose": "keras.ops.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fftnd": "keras.ops.fft.fftn({0})",
            "Ifftnd": "keras.ops.fft.ifftn({0})",
            "Rfftnd": "keras.ops.fft.rfftn({0})",
            "Irfftnd": "keras.ops.fft.irfftn({0})",
            "Fftshift": "keras.ops.fft.fftshift({0})",
            "Ifftshift": "keras.ops.fft.ifftshift({0})",
            "Dct": "tf.signal.dct({0})",
            "Idct": "tf.signal.idct({0})",
            "Mdct": "tf.signal.mdct({0})",
            "InverseMdct": "tf.signal.inverse_mdct({0})",
            "Frame": "tf.signal.frame({0})",
            "OverlapAndAdd": "tf.signal.overlap_and_add({0})",
            "Fft": "keras.ops.fft.fft({0})",
            "Rfft": "keras.ops.fft.rfft({0})",
            "Fftn": "keras.ops.fft.fftn({0})",
            "Erfinv": "keras.ops.erfinv({0})",
            "NanToNum": "keras.ops.where(keras.ops.isnan({0}), {nan}, keras.ops.where(keras.ops.isinf({0}) & ({0} > 0), {posinf}, keras.ops.where(keras.ops.isinf({0}) & ({0} < 0), {neginf}, {0})))",
            "AssignVariable": "{0}",
            "StopGradient": "kops.stop_gradient({0})",
            "Resize": "kops.image.resize({0}, {size}, method={method}, antialias={antialias})",
            "AffineGrid": "kops.image.affine_grid({0}, {size}, align_corners={align_corners})",
            "GridSample": "kops.image.grid_sample({0}, {1}, mode={mode}, padding_mode={padding_mode}, align_corners={align_corners})",
            "DrawBoundingBoxes": "{0}",
            "RgbToYiq": "kops.image.rgb_to_yiq({0})",
            "YiqToRgb": "kops.image.yiq_to_rgb({0})",
            "RgbToYuv": "kops.image.rgb_to_yuv({0})",
            "YuvToRgb": "kops.image.yuv_to_rgb({0})",
            "Ifft": "kops.fft.ifft({0})",
            "Fft2d": "kops.fft.fft2({0})",
            "Ifft2d": "kops.fft.ifft2({0})",
            "Rfft2d": "kops.fft.rfft2({0})",
            "Irfft": "kops.fft.irfft({0})",
            "Irfft2d": "kops.fft.irfft2({0})",
            "ReadVariable": "{0}",
            "Transpose": KerasTensorManipulator.format_transpose(kwargs),
        }

    def _emit_input_assignment(
        self, var_name: str, node: IRNode, input_prefix: str, input_idx: int
    ) -> None:
        """Evaluate emit input assignment.

        Args:
            var_name (str): Argument var_name
            node (IRNode): Argument node
            input_prefix (str): Argument input_prefix
            input_idx (int): Argument input_idx
        """
        self.add_line(KerasSignatureBuilder.get_input_assignment(var_name, node))
        self.keras_input_vars.append(var_name)

    def _emit_body_return(self, returns: list[str]) -> None:
        """Function docstring.

        Args:
        returns: Arg.
        """
        pass

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Evaluate emit output assignment.

        Args:
            node (IRNode): Argument node
            input_vars (list[str]): Argument input_vars
            returns (str): Argument returns
        """
        self.keras_output_vars.extend(input_vars)

    def _generate_file_header(self) -> list[str]:
        """Function docstring."""
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Function docstring."""
        import os

        tmpl_path = os.path.join(os.path.dirname(__file__), "keras_prefix.py.tmpl")
        with open(tmpl_path) as f:
            keras_prefix_template = f.read()
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
            *keras_prefix_template.split("\n"),
        ]

    def _generate_function_signature(self) -> None:
        """Function docstring."""
        self.indent_level = 0
        self.add_line("def get_model():")
        self.keras_input_vars = []
        self.keras_output_vars = []
        self.indent_level += 1

    def _traverse_ir_graph(self) -> None:
        """Function docstring."""
        self._generate_body()

    def _generate_return_block(self) -> None:
        """Function docstring."""
        self.add_line(
            KerasSignatureBuilder.get_return_block(self.keras_input_vars, self.keras_output_vars)
        )
