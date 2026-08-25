# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Einstein summation operations."""

import re
from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class EinsumLexer:
    """Lexer for Einsum equations."""

    @staticmethod
    def parse_equation_sides(equation: str) -> tuple[str, str]:
        """Evaluate parse_equation_sides operation.

        Args:
        equation (str): The equation parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        equation: object = equation.replace(" ", "")
        if "->" in equation:
            in_subs, out_sub = equation.split("->")
        else:
            in_subs: object = equation
            counts: dict[str, int] = {}
            for char in in_subs.replace(",", "").replace(".", ""):
                counts[char] = counts.get(char, 0) + 1
            out_sub: object = "".join(sorted([c for c, count in counts.items() if count == 1]))
            if "..." in in_subs:
                out_sub: object = "..." + out_sub
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
        in_parts: object = in_subs.split(",")
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
        """Evaluate _validate_ellipsis_count operation.

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
        """Evaluate _count_hidden_dims operation.

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
        num_named: object = left_len + right_len
        num_bcast: object = shape_len - num_named
        if num_bcast < 0:
            raise ValueError(f"Shape {shape} cannot match subscript {part}")
        return num_bcast

    @staticmethod
    def _combine_broadcast_shapes(broadcast_shape: Optional[tuple[int, ...]], bcast_dims: tuple[int, ...]) -> tuple[int, ...]:
        """Evaluate _combine_broadcast_shapes operation.

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
        new_shape: object = []
        for s1, s2 in zip(broadcast_shape, bcast_dims):
            if s1 != s2 and s1 != 1 and s2 != 1:
                raise ValueError("Ellipsis shapes cannot be broadcast")
            new_shape.append(max(s1, s2))
        return tuple(new_shape)

    @staticmethod
    def _handle_ellipsis(part: str, shape: tuple[int, ...], broadcast_shape: Optional[tuple[int, ...]]) -> tuple[str, tuple[int, ...], Optional[tuple[int, ...]]]:
        """Evaluate _handle_ellipsis operation.

        Args:
        part (str): The part parameter.
        shape (object): The shape parameter.
        broadcast_shape (object): The broadcast_shape parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        EinsumPlanner._validate_ellipsis_count(part, shape)
        parts_str: object = part.split("...")
        left_part: object = parts_str[0]
        right_part: object = parts_str[1]

        num_bcast: object = EinsumPlanner._count_hidden_dims(len(left_part), len(right_part), len(shape), part, shape)

        bcast_dims: object = shape[len(left_part) : len(left_part) + num_bcast]
        broadcast_shape: object = EinsumPlanner._combine_broadcast_shapes(broadcast_shape, bcast_dims)

        named_shape: object = shape[: len(left_part)] + shape[len(left_part) + num_bcast :]
        part_chars: object = left_part + right_part
        return part_chars, named_shape, broadcast_shape

    @staticmethod
    def _parse_named_part(part: str, shape: tuple[int, ...], axis_map: dict[str, int]) -> None:
        """Evaluate _parse_named_part operation.

        Args:
            part (str): The part parameter.
            shape (tuple): The shape parameter.
            axis_map (dict): The axis_map parameter.
        """
        parsed_part: object = ParsedEquationPart(part, shape)
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
        """Evaluate _parse_ellipsis_part operation.

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
        """Evaluate build_axis_size_map operation.

        Args:
        in_subs (str): The in_subs parameter.
        shapes (object): The shapes parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        in_parts: object = in_subs.split(",")
        axis_map: dict[str, int] = {}
        broadcast_shape: Optional[tuple[int, ...]] = None

        for part, shape in zip(in_parts, shapes):
            if "..." in part:
                broadcast_shape: object = EinsumPlanner._parse_ellipsis_part(part, shape, axis_map, broadcast_shape)
            else:
                EinsumPlanner._parse_named_part(part, shape, axis_map)

        return axis_map, broadcast_shape

    @staticmethod
    def _compute_output_shape_with_ellipsis(
        parts: list[str],
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> list[int]:
        """Evaluate _compute_output_shape_with_ellipsis operation.

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
        """Evaluate _resolve_chars operation.

        Args:
        out_sub (str): The out_sub parameter.
        axis_map (object): The axis_map parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        out_shape: object = []
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
        """Evaluate compute_output_shape operation.

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

        parts: object = out_sub.split("...")
        if len(parts) == MAGIC_VAL_2:
            out_shape: object = EinsumPlanner._compute_output_shape_with_ellipsis(parts, axis_map, broadcast_shape)
        else:
            out_shape: object = EinsumPlanner._resolve_chars(out_sub, axis_map)

        return tuple(out_shape)


class EinsumEquationParser:
    """Parser for Einsum equations using Lexer, Validator, and Planner."""

    @staticmethod
    def parse_equation_sides(equation: str) -> tuple[str, str]:
        """Evaluate parse_equation_sides operation.

        Args:
            equation (str): The equation parameter.

        Returns:
            tuple: Result.
        """
        return EinsumLexer.parse_equation_sides(equation)

    @staticmethod
    def build_axis_size_map(in_subs: str, shapes: list[tuple[int, ...]]) -> tuple[dict[str, int], Optional[tuple[int, ...]]]:
        """Evaluate build_axis_size_map operation.

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
        """Evaluate _resolve_chars operation.

        Args:
            out_sub (str): The out_sub parameter.
            axis_map (dict): The axis_map parameter.

        Returns:
            list: Result.
        """
        out_shape: object = []
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
        """Evaluate compute_output_shape operation.

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
        shapes (object): The shapes parameter.

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
    def _extract_equation(args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[str, tuple[object, ...]]:
        """Evaluate _extract_equation operation.

        Args:
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.

        Raises:
            ValueError: An exception.
        """
        equation: object = kwargs.get("equation", kwargs.get("subscripts"))
        if isinstance(equation, str):
            return equation, args
        if args and isinstance(args[0], str):
            return str(args[0]), args[1:]
        raise ValueError("Einsum requires an 'equation' string attribute.")

    @staticmethod
    def _extract_shapes(args: tuple[object, ...]) -> Optional[list[tuple[int, ...]]]:
        """Evaluate _extract_shapes operation.

        Args:
        args (object): The args parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        shapes: list[tuple[int, ...]] = []
        for arg in args:
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
        *args: object,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        equation, remaining_args = self._extract_equation(args, kwargs)
        shapes: object = self._extract_shapes(remaining_args)
        if shapes is None:
            return ()

        in_subs, out_sub = EinsumEquationParser.parse_equation_sides(equation)
        dim_map, ellipsis_shape = EinsumEquationParser.build_axis_size_map(in_subs, shapes)
        return EinsumEquationParser.compute_output_shape(out_sub, dim_map, ellipsis_shape)
