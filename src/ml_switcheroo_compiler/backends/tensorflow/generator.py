# ruff: noqa: E501
"""TensorFlow Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.tensorflow.tensorflow_mixins import TensorFlowControlFlowMixin, TensorFlowMathMixin


@register_backend("tensorflow")
class TensorFlowCodeGenerator(TensorFlowMathMixin, TensorFlowControlFlowMixin, BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

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

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "tf"

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        """Format the zeros like configuration or node into a backend-specific string.

        Args:
            op (str): Required parameter for op.
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        res = f"tf.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: object) -> str:
        """Format the full configuration or node into a backend-specific string.

        Args:
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        res = "tf.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs: object) -> str:
        """Format the transpose configuration or node into a backend-specific string.

        Args:
            kwargs (object): Required parameter for kwargs.

        Returns:
            str: The evaluated or processed output.
        """
        if "axes" in kwargs:
            return "tf.transpose({0}, perm={axes})"
        return "tf.transpose({0})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate RaggedDot."""
        return f"tf_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Extra arguments.

        Returns:
            str: The code string.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"tf.einsum('{eq}', {args_str})"

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "tf.math"

    def _get_creation_ops(self, kwargs: dict) -> dict[str, str]:
        """Retrieve the creation ops property or mapping.

        Args:
            kwargs (dict): Required parameter for kwargs.

        Returns:
            dict: The evaluated or processed output.
        """
        return {
            "Arange": "tf.range({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
        }

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        ops = super().get_ops_map(kwargs)
        ops["Beta"] = "tf.random.gamma({shape}, alpha={1}) / (tf.random.gamma({shape}, alpha={1}) + tf.random.gamma({shape}, alpha={2}))"
        ops["Dirichlet"] = "tf.random.gamma({shape}, alpha={1}) / tf.reduce_sum(tf.random.gamma({shape}, alpha={1}), axis=-1, keepdims=True)"
        ops["Gamma"] = "tf.random.gamma({shape}, alpha={1})"
        ops["RngBitGenerator"] = "tf.random.uniform({shape}, minval=0, maxval=255, dtype=tf.int32)"
        ops["RngUniform"] = "tf.random.uniform({shape}, minval={0}, maxval={1})"
        ops["Infeed"] = "{0}"
        ops["Outfeed"] = "{0}"
        ops["AxisIndex"] = "0"
        ops["AllToAll"] = "{0}"
        ops["Pmax"] = "{0}"
        ops["Pmin"] = "{0}"
        ops["PsumScatter"] = "{0}"
        ops["Pswapaxes"] = "{0}"
        ops["Ppermute"] = "{0}"
        ops["Pshuffle"] = "{0}"
        ops["CreateToken"] = "0"
        ops["WithShardingConstraint"] = "{0}"
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = tf.constant({val_repr})")

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
        return ["import tensorflow as tf\n"]

    def _generate_function_signature(self) -> None:
        """Evaluate and process the generate function signature operation.

        Returns:
            Any: The evaluated or processed output.
        """
        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs):")
        self.indent_level += 1
