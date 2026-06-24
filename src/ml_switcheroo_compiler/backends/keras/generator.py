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
            "Matmul": "keras.ops.matmul({0}, {1})",
            "Dot": "keras.ops.dot({0}, {1})",
            "BroadcastTo": "keras.ops.broadcast_to({0}, {shape})",
            "Reshape": "keras.ops.reshape({0}, {shape})",
            "TrueDivide": "keras.ops.true_divide({0}, {1})",
            "Arange": "keras.ops.arange({0})",
            "Zeros": KerasTensorManipulator.format_zeros_like("zeros", kwargs),
            "Ones": KerasTensorManipulator.format_zeros_like("ones", kwargs),
            "Full": KerasTensorManipulator.format_full(kwargs),
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
