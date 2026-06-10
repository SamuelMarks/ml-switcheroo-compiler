"""Type & Shape System for IR."""

from typing import Any, Union


from ml_switcheroo.ir.core import TensorSpec
from ml_switcheroo.shape import broadcast_shapes, matmul_shape


class SymInt:
    """Symbolic Integer to trace graphs with dynamic dimensions."""

    def __init__(self, name_or_expr: str) -> None:
        """Initialize SymInt."""
        self.expr = str(name_or_expr)

    def __add__(self, other: Union["SymInt", int]) -> "SymInt":
        """Docstring."""
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} + {other.expr})")
        return SymInt(f"({self.expr} + {other})")

    def __radd__(self, other: int) -> "SymInt":
        """Docstring."""
        return SymInt(f"({other} + {self.expr})")

    def __sub__(self, other: Union["SymInt", int]) -> "SymInt":
        """Docstring."""
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} - {other.expr})")
        return SymInt(f"({self.expr} - {other})")

    def __rsub__(self, other: int) -> "SymInt":
        """Docstring."""
        return SymInt(f"({other} - {self.expr})")

    def __mul__(self, other: Union["SymInt", int]) -> "SymInt":
        """Docstring."""
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} * {other.expr})")
        return SymInt(f"({self.expr} * {other})")

    def __rmul__(self, other: int) -> "SymInt":
        """Docstring."""
        return SymInt(f"({other} * {self.expr})")

    def __floordiv__(self, other: Union["SymInt", int]) -> "SymInt":
        """Docstring."""
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} // {other.expr})")
        return SymInt(f"({self.expr} // {other})")

    def __str__(self) -> str:
        """Docstring."""
        return str(self.expr)

    def __repr__(self) -> str:
        """Docstring."""
        return f"SymInt({self.expr})"

    def __eq__(self, other: Any) -> bool:
        """Docstring."""
        if isinstance(other, SymInt):
            return self.expr == other.expr
        return False


class SymbolicSolver:
    """Symbolic Expression Solver to validate shape consistency."""

    @staticmethod
    def is_consistent(expr1: Union[SymInt, int], expr2: Union[SymInt, int]) -> bool:
        """Check if two symbolic expressions are mathematically equivalent."""
        if isinstance(expr1, int) and isinstance(expr2, int):
            return expr1 == expr2

        sym1 = expr1.expr if isinstance(expr1, SymInt) else str(expr1)
        sym2 = expr2.expr if isinstance(expr2, SymInt) else str(expr2)

        # Naive string comparison for now since we can't use sympy
        return sym1 == sym2


class ShapeTracker:
    """Calculates exact output shapes given input TensorSpecs."""

    @staticmethod
    def infer_elementwise(inputs: list[TensorSpec]) -> tuple[Union[int, SymInt], ...]:
        """Infer shape for elementwise operations requiring broadcasting."""
        if not inputs:
            return ()

        shape = tuple(str(x) if isinstance(x, SymInt) else x for x in inputs[0].shape)
        for i in range(1, len(inputs)):
            s2 = tuple(str(x) if isinstance(x, SymInt) else x for x in inputs[i].shape)
            shape = broadcast_shapes(shape, s2)

        # Reconstruct SymInts
        out_shape = []
        for dim in shape:
            if isinstance(dim, str) and not dim.isdigit():
                out_shape.append(SymInt(dim))
            else:
                out_shape.append(int(dim))
        return tuple(out_shape)

    @staticmethod
    def infer_matmul(
        input1: TensorSpec, input2: TensorSpec
    ) -> tuple[Union[int, SymInt], ...]:
        """Infer shape for matrix multiplication."""
        s1 = tuple(str(x) if isinstance(x, SymInt) else x for x in input1.shape)
        s2 = tuple(str(x) if isinstance(x, SymInt) else x for x in input2.shape)

        out_shape_str = matmul_shape(s1, s2)

        out_shape = []
        for dim in out_shape_str:
            if isinstance(dim, str) and not dim.isdigit():
                out_shape.append(SymInt(dim))
            else:
                out_shape.append(int(dim))
        return tuple(out_shape)
