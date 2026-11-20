"""Shape, creation, and formatting visitor mixin for the MLX code generator."""

from __future__ import annotations

from collections.abc import Mapping

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int,
    coerce_int_list,
    coerce_shape,
    coerce_str,
)


class MLXShapeOpsVisitor:
    """Visitor for MLX shape operations."""

    def _format_zeros_ones(self, op: str, kwargs: Mapping[str, BackendArg]) -> str:
        """Format the MLX code for the zeros_ones operation.

        Args:
            op: The string name of the operation (e.g., "zeros" or "ones").
            kwargs: A dictionary of keyword arguments.

        Returns:
            A string representing the formatted MLX code for zeros_ones.
        """
        kw = kwargs
        raw_sh: BackendArg = kw.get("shape", ())
        shape: tuple[int | str, ...] = (
            tuple(coerce_int_list(raw_sh)) if isinstance(raw_sh, list) else coerce_shape(raw_sh)
        )
        dtype: str = coerce_str(kw.get("dtype", "float32"))
        dt: str = f"mx.{dtype}" if dtype else "None"
        return f"mx.{op}({shape}, dtype={dt})"

    def _format_full(self, kwargs: Mapping[str, BackendArg]) -> str:
        """Format the MLX code for the full operation.

        Args:
            kwargs (Mapping[str, BackendArg]): Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        kw = kwargs
        raw_sh: BackendArg = kw.get("shape", ())
        shape: tuple[int | str, ...] = (
            tuple(coerce_int_list(raw_sh)) if isinstance(raw_sh, list) else coerce_shape(raw_sh)
        )
        fill_value: float = coerce_float(kw.get("fill_value", 0.0))
        dtype: str = coerce_str(kw.get("dtype", "float32"))
        dt: str = f"mx.{dtype}" if dtype else "None"
        return f"mx.full({shape}, {fill_value}, dtype={dt})"

    def _format_transpose(self, kwargs: Mapping[str, BackendArg]) -> str:
        """Format the MLX code for the transpose operation.

        Args:
            kwargs (Mapping[str, BackendArg]): Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        axes: BackendArg = kwargs.get("axes", None)
        return f"axes={axes}" if axes is not None else ""

    def _format_random_categorical(self, input_vars: list[str], kwargs: Mapping[str, BackendArg]) -> str:
        """Format the MLX code for the random_categorical operation.

        Args:
            input_vars (list[str]): Input `input_vars` argument (input vars).
            kwargs (Mapping[str, BackendArg]): Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        num_samples: int = coerce_int(kwargs.get("num_samples", 1))
        return f"mx.random.categorical({input_vars[0]}, num_samples={num_samples})"

    def visit_Concatenate(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Concatenate operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Concatenate operation.
        """
        axis: int = coerce_int(node.attributes.get("axis", 0))
        return f"mx.concatenate({input_vars[0]}, axis={axis})"

    def visit_Stack(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Stack operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Stack operation.
        """
        axis: int = coerce_int(node.attributes.get("axis", 0))
        return f"mx.stack({input_vars[0]}, axis={axis})"

    def visit_Partition(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Partition operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Partition operation.
        """
        axis: int = coerce_int(node.attributes.get("axis", -1))
        kth: int = coerce_int(node.attributes.get("kth", 0))
        return f"mx.partition({input_vars[0]}, {kth}, axis={axis})"

    def visit_Argpartition(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Argpartition operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Argpartition operation.
        """
        axis: int = coerce_int(node.attributes.get("axis", -1))
        kth: int = coerce_int(node.attributes.get("kth", 0))
        return f"mx.argpartition({input_vars[0]}, {kth}, axis={axis})"

    def visit_Repeat(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Repeat operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Repeat operation.
        """
        repeats: int = coerce_int(node.attributes.get("repeats", 1))
        axis: int = coerce_int(node.attributes.get("axis", None))
        return f"mx.repeat({input_vars[0]}, {repeats}, axis={axis})"

    def visit_Roll(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Roll operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Roll operation.
        """
        shift: int = coerce_int(node.attributes.get("shift", 1))
        axis: int = coerce_int(node.attributes.get("axis", None))
        return f"mx.roll({input_vars[0]}, {shift}, axis={axis})"

    def visit_Tile(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Tile operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Tile operation.
        """
        reps: int = coerce_int(node.attributes.get("reps", 1))
        return f"mx.tile({input_vars[0]}, {reps})"

    def visit_TopK(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the TopK operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the TopK operation.
        """
        k: int = coerce_int(node.attributes.get("k", 1))
        axis: int = coerce_int(node.attributes.get("axis", -1))
        return f"mx.topk({input_vars[0]}, {k}, axis={axis})"

    def visit_Moveaxis(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Moveaxis operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Moveaxis operation.
        """
        source: int = coerce_int(node.attributes.get("source", 0))
        destination: int = coerce_int(node.attributes.get("destination", 1))
        return f"mx.moveaxis({input_vars[0]}, {source}, {destination})"

    def visit_RaggedDot(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the RaggedDot operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the RaggedDot operation.
        """
        return f"mx.matmul({input_vars[0]}, {input_vars[1]})"

    def visit_NanToNum(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the NanToNum operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the NanToNum operation.
        """
        nan: float = coerce_float(node.attributes.get("nan", 0.0))
        posinf: float = coerce_float(node.attributes.get("posinf", None))
        neginf: float = coerce_float(node.attributes.get("neginf", None))
        return f"mx.nan_to_num({input_vars[0]}, nan={nan}, posinf={posinf}, neginf={neginf})"

    def visit_Zeros(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Zeros operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Zeros operation.
        """
        return self._format_zeros_ones("zeros", node.attributes)

    def visit_Ones(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Ones operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Ones operation.
        """
        return self._format_zeros_ones("ones", node.attributes)

    def visit_Full(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Full operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Full operation.
        """
        return self._format_full(node.attributes)

    def visit_ConstantOfShape(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the ConstantOfShape operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the ConstantOfShape operation.
        """
        return self._format_full(node.attributes)

    def visit_Transpose(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Transpose operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Transpose operation.
        """
        args_str: str = self._format_transpose(node.attributes)
        return f"mx.transpose({input_vars[0]}, {args_str})" if args_str else f"mx.transpose({input_vars[0]})"

    def visit_RandomCategorical(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the RandomCategorical operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the RandomCategorical operation.
        """
        return self._format_random_categorical(input_vars, node.attributes)
