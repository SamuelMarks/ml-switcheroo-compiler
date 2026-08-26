# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""TensorFlow Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph


@register_backend("tensorflow")
class TensorFlowCodeGenerator(BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII"):
        """Load.

        Args:
        filepath (str): The filepath parameter.
        allow_pickle (bool): The allow_pickle parameter.
        fix_imports (bool): The fix_imports parameter.
        encoding (str): The encoding parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save(cls: type, file: str, arr, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr (object): The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import pickle

        with open(file, "wb") as f:
            pickle.dump(arr, f)

    @classmethod
    def savez(cls: type, file: str, *args, **kwds) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import pickle

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def savez_compressed(cls: type, file: str, *args, **kwds) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import gzip
        import pickle

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (object): The graph parameter.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def _format_zeros_like(self, op: str, kwargs) -> str:
        """Evaluate _format_zeros_like operation.

        Args:
        op (str): The op parameter.
        kwargs (object): The kwargs parameter.

        Returns:
        str: Result.
        """
        res = f"tf.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs) -> str:
        """Evaluate _format_full operation.

        Args:
        kwargs (object): The kwargs parameter.

        Returns:
        str: Result.
        """
        res = "tf.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs) -> str:
        """Evaluate _format_transpose operation.

        Args:
        kwargs (object): The kwargs parameter.

        Returns:
        str: Result.
        """
        if "axes" in kwargs:
            return "tf.transpose({0}, perm={axes})"
        return "tf.transpose({0})"

    def visit_RaggedDot(self, node, input_vars: list[str], **kwargs) -> str:
        """Evaluate visit_RaggedDot operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"tf_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node, input_vars: list[str], **kwargs) -> str:
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

    def generate(self) -> str:
        """Generate code using strict AST construction (CST) from a base NumPy string."""
        from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
        from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator

        gen = NumpyGenerator(self.graph)
        base_code = gen.generate()
        return transpile_source(base_code, target_framework="tensorflow")

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
        str: Result.
        """
        return "tf.math"

    def _get_creation_ops(self, kwargs) -> dict[str, str]:
        """Evaluate _get_creation_ops operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
        return {
            "Arange": "tf.range({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
        }

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate _emit_constant_assignment operation.

        Args:
            var_name (str): The var_name parameter.
            val_repr (str): The val_repr parameter.
        """
        self.add_line(f"{var_name} = tf.constant({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Evaluate _generate_file_header operation.

        Returns:
            tuple[int, ...]: Result.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Evaluate _resolve_imports operation.

        Returns:
            tuple[int, ...]: Result.
        """
        return ["import tensorflow as tf\n"]

    def _generate_function_signature(self) -> None:
        """Evaluate _generate_function_signature operation.

        Args:
        self (object): The self parameter.

        Returns:
        NoneType: Result.
        """
        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs) -> object:")
        self.indent_level += 1
