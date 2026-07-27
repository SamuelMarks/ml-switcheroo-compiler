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
        """Split equation into input and output subscripts."""
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
        """Validate input subscripts against shapes."""
        in_parts = in_subs.split(",")
        if len(in_parts) != len(shapes):
            raise ValueError(f"Equation expected {len(in_parts)} inputs but got {len(shapes)}")


@dataclass
class ParsedEquationPart:
    """Represents a parsed named part of an einsum equation."""

    chars: str
    shape: tuple[int, ...]

    def validate_length(self) -> None:
        """Validates that the string length matches the shape length."""
        if len(self.chars) != len(self.shape):
            raise ValueError(f"Shape {self.shape} cannot match subscript {self.chars}")

    def validate_characters(self) -> None:
        """Validates that the string only contains alphabetic characters."""
        if not re.match(r"^[a-zA-Z]*$", self.chars):
            raise ValueError(f"Invalid characters in einsum subscript: {self.chars}")

    def process_axis_map(self, axis_map: dict[str, int]) -> None:
        """Processes the characters and dimensions to update the axis map."""
        # Isolate the logic that identifies and handles duplicate dimension labels
        for char, dim in zip(self.chars, self.shape):
            self._check_dimension_mismatch(axis_map, char, dim)
            if dim != 1:
                axis_map[char] = dim

    def _check_dimension_mismatch(self, axis_map: dict[str, int], char: str, dim: int) -> None:
        """Checks for dimension mismatches, handling duplicates gracefully if matching."""
        if char in axis_map and axis_map[char] != dim and axis_map[char] != 1 and dim != 1:
            raise ValueError(f"Dimension mismatch for axis {char}")


class EinsumPlanner:
    """Planner for Einsum equations."""

    @staticmethod
    def _validate_ellipsis_count(part: str, shape: tuple[int, ...]) -> None:
        """Evaluate and process the validate ellipsis count operation.

        Args:
            part (str): Required parameter for part.
            shape (tuple): Required parameter for shape.

        Returns:
            Any: The evaluated or processed output.
        """
        if part.count("...") > 1:
            raise ValueError(f"Shape {shape} cannot match subscript {part}")

    @staticmethod
    def _count_hidden_dims(left_len: int, right_len: int, shape_len: int, part: str, shape: tuple[int, ...]) -> int:
        """Evaluate and process the count hidden dims operation.

        Args:
            left_len (int): Required parameter for left_len.
            right_len (int): Required parameter for right_len.
            shape_len (int): Required parameter for shape_len.
            part (str): Required parameter for part.
            shape (tuple): Required parameter for shape.

        Returns:
            int: The evaluated or processed output.
        """
        num_named = left_len + right_len
        num_bcast = shape_len - num_named
        if num_bcast < 0:
            raise ValueError(f"Shape {shape} cannot match subscript {part}")
        return num_bcast

    @staticmethod
    def _combine_broadcast_shapes(broadcast_shape: Optional[tuple[int, ...]], bcast_dims: tuple[int, ...]) -> tuple[int, ...]:
        """Evaluate and process the combine broadcast shapes operation.

        Args:
            broadcast_shape (Optional): Required parameter for broadcast_shape.
            bcast_dims (tuple): Required parameter for bcast_dims.

        Returns:
            tuple: The evaluated or processed output.
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
        """Evaluate and process the handle ellipsis operation.

        Args:
            part (str): Required parameter for part.
            shape (tuple): Required parameter for shape.
            broadcast_shape (Optional): Required parameter for broadcast_shape.

        Returns:
            tuple: The evaluated or processed output.
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
        """Parse the named part abstract syntax tree node into its semantic representation.

        Args:
            part (str): Required parameter for part.
            shape (tuple): Required parameter for shape.
            axis_map (dict): Required parameter for axis_map.

        Returns:
            Any: The evaluated or processed output.
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
        """Parse the ellipsis part abstract syntax tree node into its semantic representation.

        Args:
            part (str): Required parameter for part.
            shape (tuple): Required parameter for shape.
            axis_map (dict): Required parameter for axis_map.
            broadcast_shape (Optional): Required parameter for broadcast_shape.

        Returns:
            Optional: The evaluated or processed output.
        """
        part_chars, named_shape, new_broadcast_shape = EinsumPlanner._handle_ellipsis(part, shape, broadcast_shape)
        EinsumPlanner._parse_named_part(part_chars, named_shape, axis_map)
        return new_broadcast_shape

    @staticmethod
    def build_axis_size_map(in_subs: str, shapes: list[tuple[int, ...]]) -> tuple[dict[str, int], Optional[tuple[int, ...]]]:
        """Build a map of character to dimension size."""
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
        """Evaluate and process the compute output shape with ellipsis operation.

        Args:
            parts (list): Required parameter for parts.
            axis_map (dict): Required parameter for axis_map.
            broadcast_shape (Optional): Required parameter for broadcast_shape.

        Returns:
            list: The evaluated or processed output.
        """
        out_shape: list[int] = []
        for char in parts[0]:
            if char not in axis_map:
                pass
            out_shape.append(axis_map[char])
        if broadcast_shape is not None:
            out_shape.extend(broadcast_shape)
        for char in parts[1]:
            if char not in axis_map:
                pass
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def _resolve_chars(out_sub: str, axis_map: dict[str, int]) -> list[int]:
        """Evaluate and process the resolve chars operation.

        Args:
            out_sub (str): Required parameter for out_sub.
            axis_map (dict): Required parameter for axis_map.

        Returns:
            list: The evaluated or processed output.
        """
        out_shape = []
        for char in out_sub:
            if char not in axis_map:
                pass
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def compute_output_shape(
        out_sub: str,
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Compute the final output shape."""
        if out_sub.count("...") > 1:
            raise ValueError("Multiple ellipses in output subscript")

        parts = out_sub.split("...")
        if len(parts) == MAGIC_VAL_2:
            out_shape = EinsumPlanner._compute_output_shape_with_ellipsis(parts, axis_map, broadcast_shape)
        else:
            out_shape = EinsumPlanner._resolve_chars(out_sub, axis_map)

        return tuple(out_shape)


class EinsumEquationParser:
    """Parser for Einsum equations using Lexer, Validator, and Planner."""

    @staticmethod
    def parse_equation_sides(equation: str) -> tuple[str, str]:
        """Evaluate and process the parse equation sides operation.

        Args:
            equation (str): Required parameter for equation.

        Returns:
            tuple: The evaluated or processed output.
        """
        return EinsumLexer.parse_equation_sides(equation)

    @staticmethod
    def build_axis_size_map(in_subs: str, shapes: list[tuple[int, ...]]) -> tuple[dict[str, int], Optional[tuple[int, ...]]]:
        """Evaluate and process the build axis size map operation.

        Args:
            in_subs (str): Required parameter for in_subs.
            shapes (list): Required parameter for shapes.

        Returns:
            tuple: The evaluated or processed output.
        """
        EinsumValidator.validate_inputs(in_subs, shapes)
        return EinsumPlanner.build_axis_size_map(in_subs, shapes)

    @staticmethod
    def _resolve_chars(out_sub: str, axis_map: dict[str, int]) -> list[int]:
        """Evaluate and process the resolve chars operation.

        Args:
            out_sub (str): Required parameter for out_sub.
            axis_map (dict): Required parameter for axis_map.

        Returns:
            list: The evaluated or processed output.
        """
        out_shape = []
        for char in out_sub:
            if char not in axis_map:
                pass
            out_shape.append(axis_map[char])
        return out_shape

    @staticmethod
    def compute_output_shape(
        out_sub: str,
        axis_map: dict[str, int],
        broadcast_shape: Optional[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Evaluate and process the compute output shape operation.

        Args:
            out_sub (str): Required parameter for out_sub.
            axis_map (dict): Required parameter for axis_map.
            broadcast_shape (Optional): Required parameter for broadcast_shape.

        Returns:
            tuple: The evaluated or processed output.
        """
        return EinsumPlanner.compute_output_shape(out_sub, axis_map, broadcast_shape)

    @staticmethod
    def parse_and_infer_shape(equation: str, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Parse equation and infer output shape."""
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
        """Evaluate and process the extract equation operation.

        Args:
            args (tuple): Required parameter for args.
            kwargs (dict): Required parameter for kwargs.

        Returns:
            tuple: The evaluated or processed output.
        """
        equation = kwargs.get("equation", kwargs.get("subscripts"))
        if isinstance(equation, str):
            return equation, args
        if args and isinstance(args[0], str):
            return str(args[0]), args[1:]
        raise ValueError("Einsum requires an 'equation' string attribute.")

    @staticmethod
    def _extract_shapes(args: tuple[object, ...]) -> Optional[list[tuple[int, ...]]]:
        """Evaluate and process the extract shapes operation.

        Args:
            args (tuple): Required parameter for args.

        Returns:
            Optional: The evaluated or processed output.
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
            *args (object): Operand shapes.
            **kwargs (object): Additional keyword arguments, expects 'equation' or 'subscripts'.

        Returns:
            object: The computed shape.

        Raises:
            ValueError: If the equation is invalid or shapes do not match.
        """
        equation, remaining_args = self._extract_equation(args, kwargs)
        shapes = self._extract_shapes(remaining_args)
        if shapes is None:
            return ()

        in_subs, out_sub = EinsumEquationParser.parse_equation_sides(equation)
        dim_map, ellipsis_shape = EinsumEquationParser.build_axis_size_map(in_subs, shapes)  # type: ignore
        return EinsumEquationParser.compute_output_shape(out_sub, dim_map, ellipsis_shape)
