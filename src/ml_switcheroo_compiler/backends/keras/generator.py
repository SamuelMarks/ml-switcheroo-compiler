# ruff: noqa: E501
"""Keras backend generator that maps LogicalNodes and IR layers to Keras equivalent code representations."""

import os

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

from .keras_mixins import KerasAudioVisitor, KerasVisionVisitor


class KerasSignatureBuilder:
    """Helper for building Keras Model signatures."""

    @staticmethod
    def get_input_assignment(var_name: str, node: IRNode) -> str:
        """Generate the `keras.Input` declaration for a given input node.

        Args:
            var_name (str): The assigned variable name.
            node (IRNode): The logical input node representing the entry point.

        Returns:
            str: The code string for the Keras Input tensor.
        """
        shape_str = str(node.shape_metadata) if hasattr(node, "shape_metadata") and node.shape_metadata else "(None,)"
        return f"{var_name} = keras.Input(shape={shape_str}, name='{node.id}')"

    @staticmethod
    def get_return_block(input_vars: list[str], output_vars: list[str]) -> str:
        """Construct the `keras.Model` return block connecting inputs to outputs.

        Args:
            input_vars (list): List of model input variable names.
            output_vars (list): List of model output variable names.

        Returns:
            str: The code string that builds and returns the model.
        """
        inputs_str = ", ".join(input_vars)
        outputs_str = ", ".join(output_vars)
        return f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])"


class KerasTensorManipulator:
    """Helper for tensor manipulations."""

    @staticmethod
    def format_zeros_like(op: str, kwargs: object) -> str:
        """Evaluate and process the format zeros like operation.

        Args:
            op (str): Required parameter for op.
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        res = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_full(kwargs: object) -> str:
        """Evaluate and process the format full operation.

        Args:
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        res = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_transpose(kwargs: object) -> str:
        """Evaluate and process the format transpose operation.

        Args:
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"
        return "keras.ops.transpose({0})"


_KERAS_OP_REGISTRY = {
    "BroadcastInDim": "{0}.broadcast_in_dim({1}, {2})",
    "ConvGeneralDilated": "{0}.conv_general_dilated({1}, {2})",
    "DotGeneral": "{0}.dot_general({1}, {2})",
    "DynamicSlice": "{0}.dynamic_slice({1}, {2})",
    "DynamicUpdateSlice": "{0}.dynamic_update_slice({1}, {2})",
    "Pmean": "{0}.pmean({1})",
    "Psum": "{0}.psum({1})",
    "Infeed": "{0}",
    "Outfeed": "{0}",
    "AxisIndex": "0",
    "AllToAll": "{0}",
    "Pmax": "{0}",
    "Pmin": "{0}",
    "PsumScatter": "{0}",
    "Pswapaxes": "{0}",
    "Ppermute": "{0}",
    "Pshuffle": "{0}",
    "CreateToken": "0",
    "WithShardingConstraint": "{0}",
    "Beta": "keras.random.beta({shape}, {1}, {2})",
    "Dirichlet": "keras.random.dirichlet({shape}, {1})",
    "Gamma": "keras.random.gamma({shape}, {1})",
    "RngBitGenerator": "keras.random.randint({shape}, 0, 255)",
    "RngUniform": "keras.random.uniform({shape}, minval={0}, maxval={1})",
    "Matmul": "keras.ops.matmul({0}, {1})",
    "Trace": "tf.linalg.trace",
    "Adjoint": "tf.linalg.adjoint",
    "LuMatrixInverse": "tf.linalg.lu_matrix_inverse({0}, {1})",
    "LuReconstruct": "tf.linalg.lu_reconstruct({0}, {1})",
    "BandPart": "tf.linalg.band_part",
    "TriangularSolve": "tf.linalg.triangular_solve({0}, {1}, lower={lower}, adjoint={adjoint})",
    "TridiagonalSolve": "tf.linalg.tridiagonal_solve(({2}, {1}, {0}), {3}, diagonals_format='sequence')",
    "TridiagonalMatmul": "tf.linalg.tridiagonal_matmul(({2}, {1}, {0}), {3}, diagonals_format='sequence')",
    "CholeskySolve": "keras.ops.solve(keras.ops.matmul({0}, keras.ops.swapaxes({0}, -1, -2)), {1})",
    "TriInv": "tf.linalg.inv({0})",
    "Dot": "keras.ops.dot({0}, {1})",
    "BroadcastTo": "keras.ops.broadcast_to({0}, {shape})",
    "Reshape": "keras.ops.reshape({0}, {shape})",
    "TruncateDiv": "keras.ops.trunc(keras.ops.divide({0}, {1}))",
    "TruncateMod": "keras.ops.mod({0}, {1})",
    "TrueDivide": "keras.ops.true_divide({0}, {1})",
    "Arange": "keras.ops.arange({0})",
    "Sort": "keras.ops.sort({0}, axis={dimension})",
    "ArgSort": "keras.ops.argsort({0}, axis={dimension})",
    "Allclose": "keras.ops.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
    "Fftnd": "keras.ops.fft.fftn({0})",
    "Ifftnd": "keras.ops.fft.ifftn({0})",
    "Rfftnd": "keras.ops.fft.rfftn({0})",
    "Irfftnd": "keras.ops.fft.irfftn({0})",
    "Fftshift": "keras.ops.fft.fftshift({0})",
    "Ifftshift": "keras.ops.fft.ifftshift({0})",
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
}


@register_backend("keras")
class KerasCodeGenerator(BaseGenerator):
    """Emit Keras Functional API script from IR."""

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> object:
        """Load."""
        import numpy as np

        return np.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

    @classmethod
    def save(cls: type, file: str, arr: object, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save."""
        import numpy as np

        np.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez."""
        import numpy as np

        np.savez(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez compressed."""
        import numpy as np

        np.savez_compressed(file, *args, **kwds)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "keras"

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes the object.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.visitors.extend([*get_shared_ast_visitors(generator=self), KerasVisionVisitor(), KerasAudioVisitor()])
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def visit_ConvTranspose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ConvTranspose."""
        return f"keras_conv_transpose({input_vars[0]}, {input_vars[1]})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"keras_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"keras.ops.einsum('{eq}', {args_str})"

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "keras.ops"

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get op map."""
        ops = super().get_ops_map(kwargs)
        ops.update(_KERAS_OP_REGISTRY)
        ops["Zeros"] = KerasTensorManipulator.format_zeros_like("zeros", kwargs)
        ops["Ones"] = KerasTensorManipulator.format_zeros_like("ones", kwargs)
        ops["Full"] = KerasTensorManipulator.format_full(kwargs)
        ops["Transpose"] = KerasTensorManipulator.format_transpose(kwargs)
        return ops

    def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
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
        """Emit a body return operation into the active computation graph.

        Args:
            returns (list): Required parameter for returns.

        Returns:
            Any: The evaluated or processed output.
        """
        self.keras_output_vars.extend(returns)

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Evaluate emit output assignment.

        Args:
            node (IRNode): Argument node
            input_vars (list[str]): Argument input_vars
            returns (str): Argument returns
        """
        self.keras_output_vars.extend(input_vars)

    def _generate_file_header(self) -> list[str]:
        """Evaluate and process the generate file header operation.

        Returns:
            list: The evaluated or processed output.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Evaluate and process the resolve imports operation.

        Returns:
            list: The evaluated or processed output.
        """
        tmpl_path = os.path.join(os.path.dirname(__file__), "keras_prefix.py.tmpl")
        with open(tmpl_path) as f:
            keras_prefix_template = f.read()
        return ["import keras\n", *keras_prefix_template.split("\n")]

    def _generate_function_signature(self) -> None:
        """Evaluate and process the generate function signature operation.

        Returns:
            Any: The evaluated or processed output.
        """
        self.indent_level = 0
        self.add_line("def get_model():")
        self.keras_input_vars = []
        self.keras_output_vars = []
        self.indent_level += 1

    def _generate_return_block(self) -> None:
        """Evaluate and process the generate return block operation.

        Returns:
            Any: The evaluated or processed output.
        """
        self.add_line(KerasSignatureBuilder.get_return_block(self.keras_input_vars, self.keras_output_vars))
