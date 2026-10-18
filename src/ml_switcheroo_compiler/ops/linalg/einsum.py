"""Einstein summation operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class EinsumLexer:
    """Lexer for Einsum equations."""

    @staticmethod
    def parse_equation_sides(equation: str) -> tuple[str, str]:
        """Compute the parse equation sides tensor operation.

        Args:
        equation (str): The equation parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        equation = equation.replace(" ", "")
        if "->" in equation:
            in_subs, out_sub = equation.split("->")
        else:
            in_subs = equation
            counts: dict[str, int] = {}
            for char in in_subs.replace(",", "").replace(".", ""):
                counts[char] = counts.get(char, 0) + 1
            out_sub = "".join(sorted([c for c, count in counts.items() if count == 1]))
            if "..." in in_subs:
                out_sub = "..." + out_sub
        return in_subs, out_sub


class EinsumValidator:
    """Validator for Einsum equations."""

    @staticmethod
    def validate_inputs(in_subs: str, shapes: list[tuple[int, ...]]) -> None:
        """Validate input subscripts against shapes.

        Args:
            in_subs (str): The in_subs parameter.
            shapes (list): The shapes parameter.

        Raises:
            ValueError: An exception.
        """
        in_parts = in_subs.split(",")
        if len(in_parts) != len(shapes):
            raise ValueError(f"Equation expected {len(in_parts)} inputs but got {len(shapes)}")


@dataclass
class ParsedEquationPart:
    """Represents a parsed named part of an einsum equation."""

    chars: str
    shape: tuple[int, ...]

    def validate_length(self) -> None:
        """Validate that the string length matches the shape length.

        Raises:
            ValueError: An exception.
        """
        if len(self.chars) != len(self.shape):
            raise ValueError(f"Shape {self.shape} cannot match subscript {self.chars}")

    def validate_characters(self) -> None:
        """Validate that the string only contains alphabetic characters.

        Raises:
            ValueError: An exception.
        """
        if not re.match(r"^[a-zA-Z]*$", self.chars):
            raise ValueError(f"Invalid characters in einsum subscript: {self.chars}")

    def process_axis_map(self, axis_map: dict[str, int]) -> None:
        """Process the characters and dimensions to update the axis map.

        Args:
            axis_map (dict): The axis_map parameter.
        """
        # Isolate the logic that identifies and handles duplicate dimension labels
        for char, dim in zip(self.chars, self.shape):
            self._check_dimension_mismatch(axis_map, char, dim)
            if dim != 1:
                axis_map[char] = dim

    def _check_dimension_mismatch(self, axis_map: dict[str, int], char: str, dim: int) -> None:
        """Check for dimension mismatches, handling duplicates gracefully if matching.

        Args:
            axis_map (dict): The axis_map parameter.
            char (str): The char parameter.
            dim (int): The dim parameter.

        Raises:
            ValueError: An exception.
        """
        if char in axis_map and axis_map[char] != dim and axis_map[char] != 1 and dim != 1:
            raise ValueError(f"Dimension mismatch for axis {char}")


class EinsumPlanner:
    """Planner for Einsum equations."""

    @staticmethod
    def _validate_ellipsis_count(part: str, shape: tuple[int, ...]) -> None:
        """Compute the validate ellipsis count tensor operation.

        Args:
            part (str): The part parameter.
            shape (tuple): The shape parameter.

        Raises:
            ValueError: An exception.
        """
        if part.count("...") > 1:
            raise ValueError(f"Shape {shape} cannot match subscript {part}")

    @staticmethod
    def _count_hidden_dims(left_len: int, right_len: int, shape_len: int, part: str, shape: tuple[int, ...]) -> int:
        """Compute the count hidden dims tensor operation.

        Args:
            left_len (int): The left_len parameter.
            right_len (int): The right_len parameter.
            shape_len (int): The shape_len parameter.
            part (str): The part parameter.
            shape (tuple): The shape parameter.

        Returns:
            int: Result.

        Raises:
            ValueError: An exception.
        """
        num_named = left_len + right_len
        num_bcast = shape_len - num_named
        if num_bcast < 0:
            raise ValueError(f"Shape {shape} cannot match subscript {part}")
        return num_bcast

    @staticmethod
    def _combine_broadcast_shapes(broadcast_shape: Optional[tuple[int, ...]], bcast_dims: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the combine broadcast shapes tensor operation.

        Args:
            broadcast_shape (Optional): The broadcast_shape parameter.
            bcast_dims (tuple): The bcast_dims parameter.

        Returns:
            tuple: Result.

        Raises:
            ValueError: An exception.
        """
        if broadcast_shape is None:
            return bcast_dims
        new_shape = []
        for s1, s2 in zip(broadcast_shape, bcast_dims):
            if s1 != s2 and s1 != 1 and s2 != 1:
                raise ValueError("Ellipsis shapes cannot be broadcast")
            new_shape.append(max(s1, s2))
        return tuple(new_shape)

    @staticmethod
    def _handle_ellipsis(part: str, shape: tuple[int, ...], broadcast_shape: Optional[tuple[int, ...]]) -> tuple[str, tuple[int, ...], Optional[tuple[int, ...]]]:
        """Compute the handle ellipsis tensor operation.

        Args:
        part (str): The part parameter.
        shape (tuple[int, ...]): The shape parameter.
        broadcast_shape (tuple[int, ...]): The broadcast_shape parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        EinsumPlanner._validate_ellipsis_count(part, shape)
        parts_str = part.split("...")
        left_part = parts_str[0]
        right_part = parts_str[1]

        num_bcast = EinsumPlanner._count_hidden_dims(len(left_part), len(right_part), len(shape), part, shape)

        bcast_dims = shape[len(left_part) : len(left_part) + num_bcast]
        broadcast_shape = EinsumPlanner._combine_broadcast_shapes(broadcast_shape, bcast_dims)

        named_shape = shape[: len(left_part)] + shape[len(left_part) + num_bcast :]
        part_chars = left_part + right_part
        return part_chars, named_shape, broadcast_shape

    @staticmethod
    def _parse_named_part(part: str, shape: tuple[int, ...], axis_map: dict[str, int]) -> None:
        """Compute the parse named part tensor operation.

        Args:
            part (str): The part parameter.
            shape (tuple): The shape parameter.
            axis_map (dict): The axis_map parameter.
        """
        parsed_part = ParsedEquationPart(part, shape)
        parsed_part.validate_length()
        parsed_part.validate_characters()
        parsed_part.process_axis_map(axis_map)

    @staticmethod
    def _parse_ellipsis_part(
        part: str,
        shape: tuple[int, ...],
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> Optional[tuple[int, ...]]:
        """Compute the parse ellipsis part tensor operation.

        Args:
            part (str): The part parameter.
            shape (tuple): The shape parameter.
            axis_map (dict): The axis_map parameter.
            broadcast_shape (Optional): The broadcast_shape parameter.

        Returns:
            Optional: Result.
        """
        part_chars, named_shape, new_broadcast_shape = EinsumPlanner._handle_ellipsis(part, shape, broadcast_shape)
        EinsumPlanner._parse_named_part(part_chars, named_shape, axis_map)
        return new_broadcast_shape

    @staticmethod
    def build_axis_size_map(in_subs: str, shapes: list[tuple[int, ...]]) -> tuple[dict[str, int], Optional[tuple[int, ...]]]:
        """Compute the build axis size map tensor operation.

        Args:
        in_subs (str): The in_subs parameter.
        shapes (list[tuple[int, ...]]): The shapes parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        in_parts = in_subs.split(",")
        axis_map: dict[str, int] = {}
        broadcast_shape: Optional[tuple[int, ...]] = None

        for part, shape in zip(in_parts, shapes):
            if "..." in part:
                broadcast_shape = EinsumPlanner._parse_ellipsis_part(part, shape, axis_map, broadcast_shape)
            else:
                EinsumPlanner._parse_named_part(part, shape, axis_map)

        return axis_map, broadcast_shape

    @staticmethod
    def _compute_output_shape_with_ellipsis(
        parts: list[str],
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> list[int]:
        """Compute the compute output shape with ellipsis tensor operation.

        Args:
            parts (list): The parts parameter.
            axis_map (dict): The axis_map parameter.
            broadcast_shape (Optional): The broadcast_shape parameter.

        Returns:
            list: Result.
        """
        out_shape: list[int] = []
        for char in parts[0]:
            if char not in axis_map:
                continue
            out_shape.append(axis_map[char])
        if broadcast_shape is not None:
            out_shape.extend(broadcast_shape)
        for char in parts[1]:
            if char not in axis_map:
                continue
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def _resolve_chars(out_sub: str, axis_map: dict[str, int]) -> list[int]:
        """Compute the resolve chars tensor operation.

        Args:
        out_sub (str): The out_sub parameter.
        axis_map (dict[str, int]): The axis_map parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        out_shape = []
        for char in out_sub:
            if char not in axis_map:
                continue
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def compute_output_shape(
        out_sub: str,
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Compute the compute output shape tensor operation.

        Args:
            out_sub (str): The out_sub parameter.
            axis_map (dict): The axis_map parameter.
            broadcast_shape (Optional): The broadcast_shape parameter.

        Returns:
            tuple: Result.

        Raises:
            ValueError: An exception.
        """
        if out_sub.count("...") > 1:
            raise ValueError("Multiple ellipses in output subscript")

        parts = out_sub.split("...")
        if len(parts) == BINARY_OP_ARITY:
            out_shape = EinsumPlanner._compute_output_shape_with_ellipsis(parts, axis_map, broadcast_shape)
        else:
            out_shape = EinsumPlanner._resolve_chars(out_sub, axis_map)

        return tuple(out_shape)


class EinsumEquationParser:
    """Parser for Einsum equations using Lexer, Validator, and Planner."""

    @staticmethod
    def parse_equation_sides(equation: str) -> tuple[str, str]:
        """Compute the parse equation sides tensor operation.

        Args:
            equation (str): The equation parameter.

        Returns:
            tuple: Result.
        """
        return EinsumLexer.parse_equation_sides(equation)

    @staticmethod
    def build_axis_size_map(in_subs: str, shapes: list[tuple[int, ...]]) -> tuple[dict[str, int], Optional[tuple[int, ...]]]:
        """Compute the build axis size map tensor operation.

        Args:
            in_subs (str): The in_subs parameter.
            shapes (list): The shapes parameter.

        Returns:
            tuple: Result.
        """
        EinsumValidator.validate_inputs(in_subs, shapes)
        return EinsumPlanner.build_axis_size_map(in_subs, shapes)

    @staticmethod
    def _resolve_chars(out_sub: str, axis_map: dict[str, int]) -> list[int]:
        """Compute the resolve chars tensor operation.

        Args:
            out_sub (str): The out_sub parameter.
            axis_map (dict): The axis_map parameter.

        Returns:
            list: Result.
        """
        out_shape = []
        for char in out_sub:
            if char not in axis_map:
                continue
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def compute_output_shape(
        out_sub: str,
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Compute the compute output shape tensor operation.

        Args:
            out_sub (str): The out_sub parameter.
            axis_map (dict): The axis_map parameter.
            broadcast_shape (Optional): The broadcast_shape parameter.

        Returns:
            tuple: Result.
        """
        return EinsumPlanner.compute_output_shape(out_sub, axis_map, broadcast_shape)

    @staticmethod
    def parse_and_infer_shape(equation: str, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Parse equation and infer output shape.

        Args:
        equation (str): The equation parameter.
        shapes (list[tuple[int, ...]]): The shapes parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        in_subs, out_sub = EinsumLexer.parse_equation_sides(equation)
        EinsumValidator.validate_inputs(in_subs, shapes)
        axis_map, broadcast_shape = EinsumPlanner.build_axis_size_map(in_subs, shapes)
        return EinsumPlanner.compute_output_shape(out_sub, axis_map, broadcast_shape)


@register_op("Einsum")
class Einsum(OpDef):
    """Einstein summation operator.

    Evaluates the Einstein summation convention on the operands
    """

    @staticmethod
    def _extract_equation(
        args: tuple[str | tuple[int, ...] | Tensor | None, ...],
        kwargs: dict[str, str | tuple[int, ...] | Tensor | None],
    ) -> tuple[str, tuple[str | tuple[int, ...] | Tensor | None, ...]]:
        """Compute the extract equation tensor operation.

        Args:
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.

        Raises:
            ValueError: An exception.
        """
        equation = kwargs.get("equation", kwargs.get("subscripts"))
        if isinstance(equation, str):
            return equation, args
        if args and isinstance(args[0], str):
            return str(args[0]), args[1:]
        raise ValueError("Einsum requires an 'equation' string attribute.")

    @staticmethod
    def _extract_shapes(args: TensorOrScalar) -> Optional[list[tuple[int, ...]]]:
        """Compute the extract shapes tensor operation.

        Args:
        args (TensorOrScalar): The args parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        shapes: list[tuple[int, ...]] = []
        for arg in getattr(args, "__iter__")():
            if arg is None:
                continue
            if not isinstance(arg, tuple):
                return None
            shapes.append(arg)
        if not shapes:
            return None
        return shapes

    def infer_shape(
        self,
        *args: EagerTensorProtocol,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        equation, remaining_args = getattr(self, "_extract_equation")(args, kwargs)
        shapes = self._extract_shapes(remaining_args)
        if shapes is None:
            return ()

        in_subs, out_sub = EinsumEquationParser.parse_equation_sides(equation)
        dim_map, ellipsis_shape = EinsumEquationParser.build_axis_size_map(in_subs, shapes)
        return EinsumEquationParser.compute_output_shape(out_sub, dim_map, ellipsis_shape)
