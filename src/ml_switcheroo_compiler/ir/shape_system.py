"""Type & Shape System for IR with symbolic term rewriting and constraint tracking."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Union, cast

import yaml
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY, MATRIX_RANK
from ml_switcheroo_compiler.core.errors import ShapeMismatchError

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import LogicalGraph, TensorSpec

MAX_DIMENSIONS: int = 4
MAX_RANK: int = 5


class AlgebraicRuleModel(BaseModel):
    """Pydantic schema representing a single declarative algebraic rewrite rule.

    Attributes:
        name (str): Unique rule identifier.
        category (str): Semantic category of the rule.
        pattern (str): The expression pattern to match.
        replacement (str): The rewritten target pattern.
        description (str): Human-readable explanation of the simplification rule.
    """

    name: str
    category: str
    pattern: str
    replacement: str
    description: str


class AlgebraicRuleSetModel(BaseModel):
    """Pydantic schema representing the complete set of algebraic rewrite rules.

    Attributes:
        version (str): The schema version string.
        description (str): Description of the rule set.
        rules (list[AlgebraicRuleModel]): Collection of algebraic rewrite rules.
    """

    version: str = "1.0.0"
    description: str = ""
    rules: list[AlgebraicRuleModel] = Field(default_factory=list)


def load_algebraic_rules(rules_path: str | None = None) -> AlgebraicRuleSetModel:
    """Load declarative algebraic simplification rules from a YAML configuration file.

    Args:
        rules_path (Optional[str]): Optional custom path to the YAML rules file.
            Defaults to algebraic_rules.yaml in the same directory.

    Returns:
        AlgebraicRuleSetModel: Validated Pydantic rule set model.
    """
    if rules_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(current_dir, "algebraic_rules.yaml")

    if not os.path.exists(rules_path):
        return AlgebraicRuleSetModel(version="1.0.0", description="Fallback empty rules", rules=[])

    with open(rules_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return AlgebraicRuleSetModel(version="1.0.0", description="Fallback empty rules", rules=[])
    return AlgebraicRuleSetModel.model_validate(data)


MonomialType = tuple[tuple[str, int], ...]


class Polynomial:
    """Canonical multivariate polynomial over integer coefficients for symbolic equivalence.

    Attributes:
        terms (dict[MonomialType, int]): Mapping from canonical monomials to integer coefficients.
    """

    def __init__(self, terms: dict[MonomialType, int] | None = None) -> None:
        """Initialize a canonical polynomial.

        Args:
            terms (Optional[dict[MonomialType, int]]): Dictionary mapping canonical monomials
                (tuples of (var_name, power) sorted) to non-zero integer coefficients.
        """
        self.terms: dict[MonomialType, int] = {}
        if terms:
            for mono, coeff in terms.items():
                if coeff != 0:
                    self.terms[mono] = coeff

    @classmethod
    def from_const(cls, val: int) -> Polynomial:
        """Construct a constant polynomial.

        Args:
            val (int): Constant integer value.

        Returns:
            Polynomial: Resulting constant polynomial.
        """
        if val == 0:
            return cls({})
        return cls({(): val})

    @classmethod
    def from_var(cls, name: str) -> Polynomial:
        """Construct a single variable polynomial.

        Args:
            name (str): Variable identifier.

        Returns:
            Polynomial: Resulting variable polynomial.
        """
        return cls({(((name, 1),)): 1})

    def is_zero(self) -> bool:
        """Check if the polynomial is identically zero.

        Returns:
            bool: True if the polynomial has no non-zero terms.
        """
        return len(self.terms) == 0

    def is_const(self) -> bool:
        """Check if the polynomial is a constant.

        Returns:
            bool: True if polynomial contains at most a constant term.
        """
        if self.is_zero():
            return True
        return len(self.terms) == 1 and () in self.terms

    def get_const(self) -> int:
        """Return the constant integer value of the polynomial.

        Returns:
            int: The constant term value (0 if not present).
        """
        return self.terms.get((), 0)

    def add(self, other: Polynomial) -> Polynomial:
        """Add two polynomials.

        Args:
            other (Polynomial): The other polynomial.

        Returns:
            Polynomial: Sum of the two polynomials.
        """
        new_terms = dict(self.terms)
        for mono, coeff in other.terms.items():
            new_coeff = new_terms.get(mono, 0) + coeff
            if new_coeff == 0:
                new_terms.pop(mono, None)
            else:
                new_terms[mono] = new_coeff
        return Polynomial(new_terms)

    def sub(self, other: Polynomial) -> Polynomial:
        """Subtract another polynomial from self.

        Args:
            other (Polynomial): The polynomial to subtract.

        Returns:
            Polynomial: Difference of the two polynomials.
        """
        new_terms = dict(self.terms)
        for mono, coeff in other.terms.items():
            new_coeff = new_terms.get(mono, 0) - coeff
            if new_coeff == 0:
                new_terms.pop(mono, None)
            else:
                new_terms[mono] = new_coeff
        return Polynomial(new_terms)

    def mul(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials.

        Args:
            other (Polynomial): The other polynomial.

        Returns:
            Polynomial: Product of the two polynomials.
        """
        new_terms: dict[MonomialType, int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                merged_dict: dict[str, int] = {}
                for v, p in m1:
                    merged_dict[v] = merged_dict.get(v, 0) + p
                for v, p in m2:
                    merged_dict[v] = merged_dict.get(v, 0) + p
                merged_mono = tuple(sorted((v, p) for v, p in merged_dict.items() if p > 0))
                coeff = c1 * c2
                curr = new_terms.get(merged_mono, 0) + coeff
                if curr == 0:
                    new_terms.pop(merged_mono, None)
                else:
                    new_terms[merged_mono] = curr
        return Polynomial(new_terms)

    def _div_by_const(self, c: int) -> Polynomial | None:
        """Divide all terms of the polynomial by a non-zero integer constant if exact.

        Args:
            c (int): Non-zero integer divisor.

        Returns:
            Optional[Polynomial]: Quotient polynomial if all coefficients are divisible by c, else None.
        """
        if c == 0:
            return None
        new_terms: dict[MonomialType, int] = {}
        for mono, coeff in self.terms.items():
            if coeff % c != 0:
                return None
            new_terms[mono] = coeff // c
        return Polynomial(new_terms)

    def _div_by_monomial(self, div_mono: MonomialType, div_coeff: int) -> Polynomial | None:
        """Divide all terms of the polynomial by a single monomial term if exact.

        Args:
            div_mono (MonomialType): Divisor monomial powers.
            div_coeff (int): Divisor integer coefficient.

        Returns:
            Optional[Polynomial]: Quotient polynomial if exact division succeeds, else None.
        """
        new_terms: dict[MonomialType, int] = {}
        div_dict = dict(div_mono)
        for mono, coeff in self.terms.items():
            if coeff % div_coeff != 0:
                return None
            mono_dict = dict(mono)
            for var, power in div_dict.items():
                if mono_dict.get(var, 0) < power:
                    return None
                mono_dict[var] -= power
                if mono_dict[var] == 0:
                    del mono_dict[var]
            quot_mono = tuple(sorted(mono_dict.items()))
            new_terms[quot_mono] = coeff // div_coeff
        return Polynomial(new_terms)

    def div_exact(self, other: Polynomial) -> Polynomial | None:
        """Attempt exact polynomial division.

        Args:
            other (Polynomial): The divisor polynomial.

        Returns:
            Optional[Polynomial]: Quotient polynomial if division is exact, else None.
        """
        if other.is_zero():
            return None
        if self == other:
            return Polynomial.from_const(1)
        if other.is_const():
            return self._div_by_const(other.get_const())
        if len(other.terms) == 1:
            div_mono, div_coeff = next(iter(other.terms.items()))
            return self._div_by_monomial(div_mono, div_coeff)
        return None

    @staticmethod
    def _monomial_sort_key(mono: MonomialType) -> tuple[int, list[str]]:
        """Compute sort key for monomial terms by descending total degree and variable names.

        Args:
            mono (MonomialType): Monomial representation.

        Returns:
            tuple[int, list[str]]: Comparison sort key.
        """
        return (-sum(p for _, p in mono), [v for v, _ in mono])

    @staticmethod
    def _format_monomial_term(index: int, mono: MonomialType, coeff: int) -> str:
        """Format a single signed monomial term for canonical string output.

        Args:
            index (int): Position of term in the sorted polynomial sequence.
            mono (MonomialType): Canonical monomial powers.
            coeff (int): Non-zero integer coefficient.

        Returns:
            str: Formatted signed term string.
        """
        if index > 0:
            sign = " + " if coeff > 0 else " - "
        else:
            sign = "-" if coeff < 0 else ""
        abs_coeff = abs(coeff)

        if len(mono) == 0:
            return f"{sign}{abs_coeff}"

        var_str = "*".join(var if pow_val == 1 else f"{var}^{pow_val}" for var, pow_val in mono)
        part = var_str if abs_coeff == 1 else f"{abs_coeff}*{var_str}"
        return f"{sign}{part}"

    def canonical_str(self) -> str:
        """Return canonical deterministic string representation of polynomial.

        Returns:
            str: Deterministic string format with terms sorted consistently.
        """
        if self.is_zero():
            return "0"
        sorted_monos = sorted(self.terms.keys(), key=self._monomial_sort_key)
        return "".join(self._format_monomial_term(i, mono, self.terms[mono]) for i, mono in enumerate(sorted_monos))

    def __eq__(self, other: object) -> bool:
        """Evaluate equality between two polynomials.

        Args:
            other (TensorOrScalar): Other object to compare against.

        Returns:
            bool: True if both polynomials have identical canonical terms.
        """
        if not isinstance(other, Polynomial):
            return False
        return self.terms == other.terms

    def __hash__(self) -> int:
        """Compute hash of polynomial terms.

        Returns:
            int: Hash code.
        """
        return hash(tuple(sorted(self.terms.items())))

    def __str__(self) -> str:
        """Return string representation of polynomial.

        Returns:
            str: Canonical string format.
        """
        return self.canonical_str()

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"Polynomial({self.canonical_str()})"

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate polynomial expression given variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            KeyError: If a variable in the polynomial is not in env.
        """
        total = 0
        for mon, coeff in self.terms.items():
            term_val = coeff
            for var, power in mon:
                if var not in env:
                    msg = f"Variable '{var}' not bound in environment."
                    raise KeyError(msg)
                term_val *= env[var] ** power
            total += term_val
        return total


_BINARY_INT_OPS: dict[str, Callable[[int, int], int]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "**": lambda a, b: int(a**b),
    "min": min,
    "max": max,
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
}

_UNARY_INT_OPS: dict[str, Callable[[int], int]] = {
    "abs": abs,
    "-": lambda v: -v,
    "floor": lambda v: v,
    "ceil": lambda v: v,
}


def _eval_binary_int_op(op: str, v1: int, v2: int) -> int | None:
    """Evaluate a binary integer operation on two concrete integers.

    Args:
        op (str): Binary operator string.
        v1 (int): Left integer operand.
        v2 (int): Right integer operand.

    Returns:
        Optional[int]: Computed integer result if operator is recognized, else None.

    Raises:
        ZeroDivisionError: If dividing or taking modulo by zero.
    """
    if op in ("//", "/"):
        if v2 == 0:
            raise ZeroDivisionError("Division by zero in symbolic evaluation.")
        return v1 // v2
    if op == "%":
        if v2 == 0:
            raise ZeroDivisionError("Modulo by zero in symbolic evaluation.")
        return v1 % v2
    fn = _BINARY_INT_OPS.get(op)
    return fn(v1, v2) if fn is not None else None


class SymNode:
    """Base class for all symbolic expression tree nodes."""

    @classmethod
    def to_node(cls, val: int | str | SymNode | SymInt) -> SymNode:
        """Convert an int, str, SymNode, or SymInt into a SymNode.

        Args:
            val (Union[int, str, SymNode, SymInt]): Value to convert.

        Returns:
            SymNode: Structured symbolic expression node.
        """
        if isinstance(val, SymNode):
            return val
        if isinstance(val, SymInt):
            return val.node
        if isinstance(val, int):
            return SymConst(val)
        if isinstance(val, str):
            return _parse_sym_str(val)
        msg = f"Cannot convert {type(val)} to SymNode"
        raise TypeError(msg)

    def simplify(self) -> SymNode:
        """Apply algebraic simplification rules recursively.

        Returns:
            SymNode: Simplified expression tree.
        """
        return self

    def canonical(self) -> SymNode:
        """Reduce expression to canonical polynomial form.

        Returns:
            SymNode: Canonical expression tree.
        """
        poly = self.to_polynomial()
        if poly is not None:
            return _poly_to_symnode(poly)
        return self.simplify()

    def to_polynomial(self) -> Polynomial | None:
        """Convert node to canonical Polynomial representation if possible.

        Returns:
            Optional[Polynomial]: Canonical polynomial or None if non-polynomial.
        """
        return None

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate expression given variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            KeyError: If an unbound variable is encountered.
            NotImplementedError: If node type does not support evaluation.
        """
        poly = self.to_polynomial()
        if poly is not None:
            return poly.eval(env)
        simplified = self.simplify()
        if simplified is not self:
            return simplified.eval(env)
        left = getattr(self, "left", None)
        right = getattr(self, "right", None)
        op = getattr(self, "op", None)
        if isinstance(left, SymNode) and isinstance(right, SymNode) and op is not None:
            res = _eval_binary_int_op(str(op), left.eval(env), right.eval(env))
            if res is not None:
                return res
        operand = getattr(self, "operand", None)
        if isinstance(operand, SymNode) and op is not None:
            u_fn = _UNARY_INT_OPS.get(str(op))
            if u_fn is not None:
                return u_fn(operand.eval(env))
        msg = f"Evaluation not implemented for node of type {type(self).__name__}"
        raise NotImplementedError(msg)

    def free_vars(self) -> set[str]:
        """Return set of free variable names in expression.

        Returns:
            set[str]: Set of variable name strings.
        """
        return set()

    def __add__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Add two symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Operand to add.

        Returns:
            SymBinaryOp: Addition expression node.
        """
        return SymBinaryOp("+", self, SymNode.to_node(other))

    def __radd__(self, other: int | str) -> SymBinaryOp:
        """Right-add two symbolic expressions.

        Args:
            other (Union[int, str]): Left operand.

        Returns:
            SymBinaryOp: Addition expression node.
        """
        return SymBinaryOp("+", SymNode.to_node(other), self)

    def __sub__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Subtract symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Operand to subtract.

        Returns:
            SymBinaryOp: Subtraction expression node.
        """
        return SymBinaryOp("-", self, SymNode.to_node(other))

    def __rsub__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-subtract symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Left operand.

        Returns:
            SymBinaryOp: Subtraction expression node.
        """
        return SymBinaryOp("-", SymNode.to_node(other), self)

    def __mul__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Multiply symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Operand to multiply.

        Returns:
            SymBinaryOp: Multiplication expression node.
        """
        return SymBinaryOp("*", self, SymNode.to_node(other))

    def __rmul__(self, other: int | str) -> SymBinaryOp:
        """Right-multiply symbolic expressions.

        Args:
            other (Union[int, str]): Left operand.

        Returns:
            SymBinaryOp: Multiplication expression node.
        """
        return SymBinaryOp("*", SymNode.to_node(other), self)

    def __floordiv__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Floor-divide symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Divisor operand.

        Returns:
            SymBinaryOp: Floor division expression node.
        """
        return SymBinaryOp("//", self, SymNode.to_node(other))

    def __rfloordiv__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-floor-divide symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Dividend operand.

        Returns:
            SymBinaryOp: Floor division expression node.
        """
        return SymBinaryOp("//", SymNode.to_node(other), self)

    def __mod__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Modulo symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Divisor operand.

        Returns:
            SymBinaryOp: Modulo expression node.
        """
        return SymBinaryOp("%", self, SymNode.to_node(other))

    def __rmod__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-modulo symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Dividend operand.

        Returns:
            SymBinaryOp: Modulo expression node.
        """
        return SymBinaryOp("%", SymNode.to_node(other), self)

    def __pow__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Exponentiate symbolic expression.

        Args:
            other (Union[int, str, SymNode, SymInt]): Exponent operand.

        Returns:
            SymBinaryOp: Power expression node.
        """
        return SymBinaryOp("**", self, SymNode.to_node(other))

    def __rpow__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-exponentiate symbolic expression.

        Args:
            other (Union[int, str, SymNode, SymInt]): Base operand.

        Returns:
            SymBinaryOp: Power expression node.
        """
        return SymBinaryOp("**", SymNode.to_node(other), self)

    def __neg__(self) -> SymUnaryOp:
        """Negate symbolic expression.

        Returns:
            SymUnaryOp: Negated expression node.
        """
        return SymUnaryOp("-", self)

    def __abs__(self) -> SymUnaryOp:
        """Absolute value of symbolic expression.

        Returns:
            SymUnaryOp: Absolute value node.
        """
        return SymUnaryOp("abs", self)


class SymConst(SymNode):
    """Constant integer symbolic node.

    Attributes:
        value (int): Integer constant value.
    """

    def __init__(self, value: int) -> None:
        """Initialize SymConst.

        Args:
            value (int): Constant integer value.
        """
        self.value: int = int(value)

    def simplify(self) -> SymConst:
        """Simplify constant node.

        Returns:
            SymConst: Self.
        """
        return self

    def to_polynomial(self) -> Polynomial:
        """Convert constant to polynomial.

        Returns:
            Polynomial: Constant polynomial.
        """
        return Polynomial.from_const(self.value)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate constant value.

        Args:
            env (dict[str, int]): Variable bindings.

        Returns:
            int: Constant integer value.
        """
        return self.value

    def free_vars(self) -> set[str]:
        """Return free variables for constant.

        Returns:
            set[str]: Empty set.
        """
        return set()

    def __str__(self) -> str:
        """Return string representation of constant.

        Returns:
            str: String of integer value.
        """
        return str(self.value)

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymConst({self.value})"

    def __eq__(self, other: object) -> bool:
        """Check equality with constant.

        Args:
            other (TensorOrScalar): Other object.

        Returns:
            bool: True if values match.
        """
        if isinstance(other, SymConst):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return False

    def __hash__(self) -> int:
        """Return hash for constant.

        Returns:
            int: Hash integer.
        """
        return hash(self.value)


class SymVar(SymNode):
    """Symbolic variable node representing a dynamic dimension.

    Attributes:
        name (str): Identifier name of the variable.
    """

    def __init__(self, name: str) -> None:
        """Initialize SymVar.

        Args:
            name (str): Variable name.
        """
        self.name: str = str(name)

    def simplify(self) -> SymVar:
        """Simplify variable node.

        Returns:
            SymVar: Self.
        """
        return self

    def to_polynomial(self) -> Polynomial:
        """Convert variable to polynomial.

        Returns:
            Polynomial: Variable polynomial.
        """
        return Polynomial.from_var(self.name)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate variable given environment.

        Args:
            env (dict[str, int]): Environment dictionary.

        Returns:
            int: Bound value.

        Raises:
            KeyError: If variable is not bound.
        """
        if self.name not in env:
            msg = f"Variable '{self.name}' not bound in environment."
            raise KeyError(msg)
        return env[self.name]

    def free_vars(self) -> set[str]:
        """Return set containing variable name.

        Returns:
            set[str]: Set with variable name.
        """
        return {self.name}

    def __str__(self) -> str:
        """Return variable name string.

        Returns:
            str: Name.
        """
        return self.name

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymVar({self.name!r})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another variable.

        Args:
            other (TensorOrScalar): Other object.

        Returns:
            bool: True if variable names match.
        """
        if isinstance(other, SymVar):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        """Compute hash for variable.

        Returns:
            int: Hash integer.
        """
        return hash(self.name)


def _is_nonneg_const(node: SymNode) -> bool:
    """Check if a SymNode is a non-negative integer constant.

    Args:
        node (SymNode): Node to inspect.

    Returns:
        bool: True if node is a SymConst with value >= 0.
    """
    return isinstance(node, SymConst) and node.value >= 0


def _has_nonneg_offset(expr: SymNode, target: SymNode) -> bool:
    """Check if expr is of the form (target + c) or (c + target) with c >= 0.

    Args:
        expr (SymNode): Candidate additive expression.
        target (SymNode): Base expression to match against.

    Returns:
        bool: True if expr equals target plus a non-negative constant.
    """
    if not isinstance(expr, SymBinaryOp) or expr.op != "+":
        return False
    return (expr.left == target and _is_nonneg_const(expr.right)) or (expr.right == target and _is_nonneg_const(expr.left))


def _has_nonneg_decrement(expr: SymNode, target: SymNode) -> bool:
    """Check if expr is of the form (target - c) with c >= 0.

    Args:
        expr (SymNode): Candidate subtractive expression.
        target (SymNode): Base expression to match against.

    Returns:
        bool: True if expr equals target minus a non-negative constant.
    """
    return isinstance(expr, SymBinaryOp) and expr.op == "-" and expr.left == target and _is_nonneg_const(expr.right)


def _simplify_add(s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify addition identity cases.

    Args:
        s_left (SymNode): Simplified left operand.
        s_right (SymNode): Simplified right operand.

    Returns:
        Optional[SymNode]: Simplified node if an identity applies, else None.
    """
    if isinstance(s_right, SymConst) and s_right.value == 0:
        return s_left
    if isinstance(s_left, SymConst) and s_left.value == 0:
        return s_right
    return None


def _simplify_sub(s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify subtraction identity and cancellation cases.

    Args:
        s_left (SymNode): Simplified left operand.
        s_right (SymNode): Simplified right operand.

    Returns:
        Optional[SymNode]: Simplified node if a rule applies, else None.
    """
    if isinstance(s_right, SymConst) and s_right.value == 0:
        return s_left
    if s_left == s_right:
        return SymConst(0)
    if isinstance(s_left, SymBinaryOp) and s_left.op == "+":
        if s_left.right == s_right:
            return s_left.left
        if s_left.left == s_right:
            return s_left.right
    return None


def _simplify_mul(s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify multiplication identity and zero annihilation cases.

    Args:
        s_left (SymNode): Simplified left operand.
        s_right (SymNode): Simplified right operand.

    Returns:
        Optional[SymNode]: Simplified node if a rule applies, else None.
    """
    if (isinstance(s_right, SymConst) and s_right.value == 0) or (isinstance(s_left, SymConst) and s_left.value == 0):
        return SymConst(0)
    if isinstance(s_right, SymConst) and s_right.value == 1:
        return s_left
    if isinstance(s_left, SymConst) and s_left.value == 1:
        return s_right
    return None


def _simplify_floordiv_mul(s_left: SymBinaryOp, s_right: SymNode) -> SymNode | None:
    """Simplify floor division when dividend is a multiplication expression.

    Args:
        s_left (SymBinaryOp): Multiplication expression on the left.
        s_right (SymNode): Simplified divisor operand.

    Returns:
        Optional[SymNode]: Simplified node if cancellation applies, else None.
    """
    if s_left.right == s_right:
        return s_left.left
    if s_left.left == s_right:
        return s_left.right
    if isinstance(s_right, SymConst) and s_right.value != 0:
        if isinstance(s_left.right, SymConst) and s_left.right.value % s_right.value == 0:
            return SymBinaryOp("*", s_left.left, SymConst(s_left.right.value // s_right.value)).simplify()
        if isinstance(s_left.left, SymConst) and s_left.left.value % s_right.value == 0:
            return SymBinaryOp("*", s_left.right, SymConst(s_left.left.value // s_right.value)).simplify()
    return None


def _simplify_floordiv(s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify floor division identity and factor cancellation cases.

    Args:
        s_left (SymNode): Simplified dividend operand.
        s_right (SymNode): Simplified divisor operand.

    Returns:
        Optional[SymNode]: Simplified node if a rule applies, else None.
    """
    if isinstance(s_right, SymConst) and s_right.value == 1:
        return s_left
    if isinstance(s_left, SymConst) and s_left.value == 0:
        return SymConst(0)
    if s_left == s_right and not (isinstance(s_right, SymConst) and s_right.value == 0):
        return SymConst(1)
    if isinstance(s_left, SymBinaryOp) and s_left.op == "*":
        return _simplify_floordiv_mul(s_left, s_right)
    return None


def _simplify_mod(s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify modulo identity cases.

    Args:
        s_left (SymNode): Simplified dividend operand.
        s_right (SymNode): Simplified divisor operand.

    Returns:
        Optional[SymNode]: SymConst(0) if divisible, else None.
    """
    if (isinstance(s_right, SymConst) and s_right.value == 1) or (isinstance(s_left, SymConst) and s_left.value == 0):
        return SymConst(0)
    if s_left == s_right and not (isinstance(s_right, SymConst) and s_right.value == 0):
        return SymConst(0)
    return None


def _simplify_min_max(op: str, s_left: SymNode, s_right: SymNode) -> SymNode | None:
    """Simplify min and max idempotence, absorption, and monotonic offset rules.

    Args:
        op (str): Operator string ('min' or 'max').
        s_left (SymNode): Simplified left operand.
        s_right (SymNode): Simplified right operand.

    Returns:
        Optional[SymNode]: Simplified node if a rule applies, else None.
    """
    if s_left == s_right:
        return s_left
    if isinstance(s_left, SymConst) and isinstance(s_right, SymConst):
        return SymConst(min(s_left.value, s_right.value) if op == "min" else max(s_left.value, s_right.value))
    if isinstance(s_left, SymBinaryOp) and s_left.op == op and (s_left.left == s_right or s_left.right == s_right):
        return s_left
    if isinstance(s_right, SymBinaryOp) and s_right.op == op and (s_right.left == s_left or s_right.right == s_left):
        return s_right
    res: SymNode | None = None
    if _has_nonneg_offset(s_left, s_right) or _has_nonneg_decrement(s_right, s_left):
        res = s_right if op == "min" else s_left
    elif _has_nonneg_offset(s_right, s_left) or _has_nonneg_decrement(s_left, s_right):
        res = s_left if op == "min" else s_right
    return res


_BINARY_SIMPLIFIERS: dict[str, Callable[[SymNode, SymNode], SymNode | None]] = {
    "+": _simplify_add,
    "-": _simplify_sub,
    "*": _simplify_mul,
    "//": _simplify_floordiv,
    "%": _simplify_mod,
    "min": lambda l_node, r_node: _simplify_min_max("min", l_node, r_node),
    "max": lambda l_node, r_node: _simplify_min_max("max", l_node, r_node),
}


class SymBinaryOp(SymNode):
    """Binary operation node in symbolic expression tree.

    Attributes:
        op (str): Binary operator (+, -, *, //).
        left (SymNode): Left operand node.
        right (SymNode): Right operand node.
    """

    def __init__(self, op: str, left: SymNode, right: SymNode) -> None:
        """Initialize SymBinaryOp.

        Args:
            op (str): Operator string.
            left (SymNode): Left child node.
            right (SymNode): Right child node.
        """
        self.op: str = str(op)
        self.left: SymNode = left
        self.right: SymNode = right

    @staticmethod
    def _fold_constants(op: str, s_left: SymNode, s_right: SymNode) -> SymConst | None:
        """Fold binary arithmetic operations on two constant operands.

        Args:
            op (str): Binary operator string.
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            Optional[SymConst]: Folded constant node if applicable, else None.
        """
        if not (isinstance(s_left, SymConst) and isinstance(s_right, SymConst)):
            return None
        v1, v2 = s_left.value, s_right.value
        if op in ("+", "-", "*"):
            return SymConst(_BINARY_INT_OPS[op](v1, v2))
        if op == "//" and v2 != 0:
            return SymConst(v1 // v2)
        return None

    def simplify(self) -> SymNode:
        """Apply algebraic rewrite rules to simplify binary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        folded = self._fold_constants(self.op, s_left, s_right)
        if folded is not None:
            return folded

        rule_fn = _BINARY_SIMPLIFIERS.get(self.op)
        if rule_fn is not None:
            res = rule_fn(s_left, s_right)
            if res is not None:
                return res

        poly = self.to_polynomial()
        if poly is not None:
            return _poly_to_symnode(poly)

        return SymBinaryOp(self.op, s_left, s_right)

    def to_polynomial(self) -> Polynomial | None:
        """Convert binary operation to canonical Polynomial representation.

        Returns:
            Optional[Polynomial]: Resulting polynomial or None if irreducible.
        """
        p_left = self.left.to_polynomial()
        p_right = self.right.to_polynomial()
        if p_left is None or p_right is None:
            return None

        if self.op == "+":
            return p_left.add(p_right)
        if self.op == "-":
            return p_left.sub(p_right)
        if self.op == "*":
            return p_left.mul(p_right)
        if self.op == "//":
            return p_left.div_exact(p_right)
        return None

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate binary operation given variable bindings.

        Args:
            env (dict[str, int]): Environment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            ZeroDivisionError: If dividing or taking modulo by zero.
            ValueError: If operator is unsupported.
        """
        v1 = self.left.eval(env)
        v2 = self.right.eval(env)
        res = _eval_binary_int_op(self.op, v1, v2)
        if res is not None:
            return res
        raise ValueError(f"Unsupported operator '{self.op}'.")

    def free_vars(self) -> set[str]:
        """Return free variables of both operands.

        Returns:
            set[str]: Union of free variables.
        """
        return self.left.free_vars() | self.right.free_vars()

    def __str__(self) -> str:
        """Format binary operation as a string with parentheses.

        Returns:
            str: Parenthesized string expression.
        """
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymBinaryOp({self.op!r}, {self.left!r}, {self.right!r})"

    def __eq__(self, other: object) -> bool:
        """Check structural equality with another binary operation node.

        Args:
            other (TensorOrScalar): Other object.

        Returns:
            bool: True if other is a SymBinaryOp with identical op, left, and right operands.
        """
        if not isinstance(other, SymBinaryOp):
            return NotImplemented
        return self.op == other.op and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        """Compute hash for binary operation.

        Returns:
            int: Hash integer based on op, left, and right operands.
        """
        return hash((self.op, self.left, self.right))


class SymUnaryOp(SymNode):
    """Unary operation node in symbolic expression tree (abs, ceil, floor, neg).

    Attributes:
        op (str): Unary operator name.
        operand (SymNode): Target operand node.
    """

    def __init__(self, op: str, operand: SymNode) -> None:
        """Initialize SymUnaryOp.

        Args:
            op (str): Operator string ('abs', 'ceil', 'floor', '-').
            operand (SymNode): Child operand node.
        """
        self.op = str(op)
        self.operand = operand

    def simplify(self) -> SymNode:
        """Simplify unary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_op = self.operand.simplify()
        if isinstance(s_op, SymConst):
            v = s_op.value
            if self.op == "abs":
                return SymConst(abs(v))
            if self.op in ("ceil", "floor"):
                return SymConst(v)
            if self.op == "-":
                return SymConst(-v)
        return SymUnaryOp(self.op, s_op)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate unary operation with variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer value.

        Raises:
            ValueError: If operator is unsupported.
        """
        val = self.operand.eval(env)
        if self.op == "abs":
            return abs(val)
        if self.op in ("ceil", "floor"):
            return val
        if self.op == "-":
            return -val
        raise ValueError(f"Unsupported unary operator '{self.op}'.")

    def free_vars(self) -> set[str]:
        """Return free variables of the operand.

        Returns:
            set[str]: Set of variable names.
        """
        return self.operand.free_vars()

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: Operator and operand string.
        """
        if self.op == "-":
            return f"-({self.operand})"
        return f"{self.op}({self.operand})"

    def __repr__(self) -> str:
        """Return debugging string representation.

        Returns:
            str: Debugging string.
        """
        return f"SymUnaryOp({self.op}, {self.operand!r})"


class SymPiecewise(SymNode):
    """Piecewise conditional symbolic node.

    Attributes:
        cases (Sequence[tuple[Callable[[dict[str, int]], bool] | SymNode | bool, SymNode]]): Sequence of (condition, expr) pairs.
        default (SymNode): Default fallback expression node.
    """

    def __init__(
        self,
        cases: Sequence[tuple[Callable[[dict[str, int]], bool] | SymNode | bool, SymNode]],
        default: SymNode,
    ) -> None:
        """Initialize SymPiecewise.

        Args:
            cases (Sequence[tuple[Callable[[dict[str, int]], bool] | SymNode | bool, SymNode]]): List of (condition_fn_or_node, expr_node).
            default (SymNode): Fallback node when no conditions are met.
        """
        self.cases = list(cases)
        self.default = default

    def simplify(self) -> SymNode:
        """Simplify piecewise expression.

        Returns:
            SymNode: Simplified expression node.
        """
        simplified_cases = []
        for cond, expr in self.cases:
            simplified_cases.append((cond, expr.simplify()))
        return SymPiecewise(simplified_cases, self.default.simplify())

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate piecewise expression given environment bindings.

        Args:
            env (dict[str, int]): Variable bindings dictionary.

        Returns:
            int: Evaluated integer result from the first matching branch or default.
        """
        for cond, expr in self.cases:
            is_met = False
            if callable(cond):
                is_met = bool(cond(env))
            elif isinstance(cond, SymNode):
                is_met = bool(cond.eval(env))
            elif isinstance(cond, bool):
                is_met = cond
            if is_met:
                return expr.eval(env)
        return self.default.eval(env)

    def free_vars(self) -> set[str]:
        """Return union of free variables across all branches.

        Returns:
            set[str]: Set of variable names.
        """
        vars_set = set(self.default.free_vars())
        for cond, expr in self.cases:
            if isinstance(cond, SymNode):
                vars_set.update(cond.free_vars())
            vars_set.update(expr.free_vars())
        return vars_set

    def __eq__(self, other: object) -> bool:
        """Check structural equality.

        Args:
            other (TensorOrScalar): Other object.

        Returns:
            bool: True if cases and default match.
        """
        if isinstance(other, SymPiecewise):
            return self.cases == other.cases and self.default == other.default
        return False

    def __hash__(self) -> int:
        """Compute hash code.

        Returns:
            int: Hash integer.
        """
        return hash((tuple(self.cases), self.default))


def _build_monomial_node(mono: MonomialType, abs_coeff: int) -> SymNode:
    """Build a product SymNode for a single monomial with positive coefficient.

    Args:
        mono (MonomialType): Canonical monomial powers.
        abs_coeff (int): Positive integer coefficient.

    Returns:
        SymNode: Product expression tree for the monomial term.
    """
    factors: list[SymNode] = []
    if abs_coeff != 1:
        factors.append(SymConst(abs_coeff))
    for var, power in mono:
        factors.extend(SymVar(var) for _ in range(power))
    if not factors:
        factors.append(SymConst(abs_coeff))

    term_node = factors[0]
    for factor in factors[1:]:
        term_node = SymBinaryOp("*", term_node, factor)
    return term_node


def _poly_to_symnode(poly: Polynomial) -> SymNode:
    """Convert a canonical Polynomial back into a simplified SymNode expression tree.

    Args:
        poly (Polynomial): Canonical polynomial instance.

    Returns:
        SymNode: Equivalent structured symbolic expression tree.
    """
    if poly.is_zero():
        return SymConst(0)
    if poly.is_const():
        return SymConst(poly.get_const())

    sorted_monos = sorted(poly.terms.keys(), key=Polynomial._monomial_sort_key)
    first_mono = sorted_monos[0]
    first_coeff = poly.terms[first_mono]
    if first_coeff < 0:
        res_node: SymNode = SymBinaryOp("-", SymConst(0), _build_monomial_node(first_mono, -first_coeff))
    else:
        res_node = _build_monomial_node(first_mono, first_coeff)

    for mono in sorted_monos[1:]:
        coeff = poly.terms[mono]
        op = "-" if coeff < 0 else "+"
        term_node = _build_monomial_node(mono, abs(coeff))
        res_node = SymBinaryOp(op, res_node, term_node)

    return res_node


def _scan_token(s: str, i: int) -> tuple[str | None, int]:
    """Scan a single lexical token starting at index i in expression string s.

    Args:
        s (str): Stripped expression string.
        i (int): Current scan index.

    Returns:
        tuple[Optional[str], int]: Tuple of (token_or_none, next_index).
    """
    n = len(s)
    c = s[i]
    if c.isspace():
        return None, i + 1
    if c in "()+*-":
        return c, i + 1
    if c == "/":
        step = 2 if (i + 1 < n and s[i + 1] == "/") else 1
        return "//", i + step
    if c.isdigit():
        end = i
        while end < n and s[end].isdigit():
            end += 1
        return s[i:end], end
    if c.isalpha() or c == "_":
        end = i
        while end < n and (s[end].isalnum() or s[end] == "_"):
            end += 1
        return s[i:end], end
    return None, i + 1


def _tokenize_expr(expr_str: str) -> list[str]:
    """Tokenize a symbolic expression string into component tokens.

    Args:
        expr_str (str): Input expression string.

    Returns:
        list[str]: Sequence of string tokens.
    """
    tokens: list[str] = []
    s = expr_str.strip()
    i = 0
    while i < len(s):
        tok, i = _scan_token(s, i)
        if tok is not None:
            tokens.append(tok)
    return tokens


class _SymExprParser:
    """Recursive descent parser for symbolic dimension expressions."""

    def __init__(self, tokens: list[str]) -> None:
        """Initialize _SymExprParser with a token list.

        Args:
            tokens (list[str]): Sequence of lexical tokens.
        """
        self.tokens = tokens
        self.idx = 0

    def parse_expr(self) -> SymNode:
        """Parse additive binary expressions.

        Returns:
            SymNode: Root parsed expression node.
        """
        node = self.parse_term()
        while self.idx < len(self.tokens) and self.tokens[self.idx] in ("+", "-"):
            op = self.tokens[self.idx]
            self.idx += 1
            node = SymBinaryOp(op, node, self.parse_term())
        return node

    def parse_term(self) -> SymNode:
        """Parse multiplicative binary expressions.

        Returns:
            SymNode: Root parsed term node.
        """
        node = self.parse_factor()
        while self.idx < len(self.tokens) and self.tokens[self.idx] in ("*", "//"):
            op = self.tokens[self.idx]
            self.idx += 1
            node = SymBinaryOp(op, node, self.parse_factor())
        return node

    def parse_factor(self) -> SymNode:
        """Parse factor, parenthesized expression, or unary negative.

        Returns:
            SymNode: Root parsed factor node.
        """
        if self.idx >= len(self.tokens):
            return SymConst(0)
        tok = self.tokens[self.idx]
        if tok == "(":
            self.idx += 1
            node = self.parse_expr()
            if self.idx < len(self.tokens) and self.tokens[self.idx] == ")":
                self.idx += 1
            return node
        if tok == "-":
            self.idx += 1
            return SymBinaryOp("-", SymConst(0), self.parse_factor())
        self.idx += 1
        return SymConst(int(tok)) if tok.isdigit() else SymVar(tok)


def _parse_sym_str(s: str) -> SymNode:
    """Parse an algebraic string expression into a structured SymNode.

    Args:
        s (str): String expression.

    Returns:
        SymNode: Structured symbolic expression node.
    """
    tokens = _tokenize_expr(s)
    if not tokens:
        return SymVar(s)

    parser = _SymExprParser(tokens)
    try:
        node = parser.parse_expr()
        if parser.idx == len(tokens):
            return node
    except Exception:
        return SymVar(s)
    return SymVar(s)


class SymInt:
    """Symbolic Integer to trace graphs with dynamic dimensions."""

    def __init__(self, name_or_expr: str | int | SymNode | SymInt) -> None:
        """Initialize SymInt.

        Args:
            name_or_expr (Union[str, int, SymNode, SymInt]): The name, expression string,
                integer, or underlying SymNode.
        """
        if isinstance(name_or_expr, SymInt):
            self.node: SymNode = name_or_expr.node
            self.expr: str = name_or_expr.expr
        elif isinstance(name_or_expr, SymNode):
            self.node = name_or_expr
            self.expr = str(name_or_expr)
        elif isinstance(name_or_expr, int):
            self.node = SymConst(name_or_expr)
            self.expr = str(name_or_expr)
        else:
            str_val = str(name_or_expr)
            self.node = SymNode.to_node(str_val)
            self.expr = str_val

    @property
    def name(self) -> str:
        """Return the symbolic variable identifier or expression string.

        Returns:
            str: Identifier or expression string.
        """
        return self.expr

    def simplify(self) -> SymInt:
        """Return simplified SymInt instance.

        Returns:
            SymInt: Simplified expression.
        """
        return SymInt(self.node.simplify())

    def canonical(self) -> SymInt:
        """Return canonicalized SymInt instance.

        Returns:
            SymInt: Canonical polynomial reduced expression.
        """
        return SymInt(self.node.canonical())

    def __add__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Execute add for compiler processing.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", self.node, other_node))

    def __radd__(self, other: int | str) -> SymInt:
        """Execute radd for compiler processing.

        Args:
            other (Union[int, str]): The other operand.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", other_node, self.node))

    def __sub__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Execute sub for compiler processing.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", self.node, other_node))

    def __rsub__(self, other: int | SymNode | str) -> SymInt:
        """Execute rsub for compiler processing.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", other_node, self.node))

    def __mul__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Execute mul for compiler processing.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", self.node, other_node))

    def __rmul__(self, other: int | str) -> SymInt:
        """Execute rmul for compiler processing.

        Args:
            other (Union[int, str]): The other operand.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", other_node, self.node))

    def __floordiv__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Execute floordiv for compiler processing.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", self.node, other_node))

    def __rfloordiv__(self, other: int | SymNode | str) -> SymInt:
        """Execute rfloordiv for compiler processing.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", other_node, self.node))

    def __str__(self) -> str:
        """Execute str for compiler processing.

        Returns:
            str: String expression.
        """
        return str(self.node)

    def __repr__(self) -> str:
        """Execute repr for compiler processing.

        Returns:
            str: Debugging representation.
        """
        return f"SymInt({self.node})"

    def __hash__(self) -> int:
        """Execute hash for compiler processing.

        Returns:
            int: Hash integer.
        """
        return hash(self.expr)

    def __eq__(self, other: object) -> bool:
        """Evaluate __eq__ operation using canonical polynomial consistency.

        Args:
            other (TensorOrScalar): The other object to compare.

        Returns:
            bool: True if expressions are mathematically consistent.
        """
        if isinstance(other, (SymInt, SymNode, int)):
            return SymbolicSolver.is_consistent(self, other)
        return False


class SymbolicSolver:
    """Symbolic Expression Solver to validate shape consistency."""

    @staticmethod
    def is_consistent(
        expr1: SymInt | SymNode | int | str,
        expr2: SymInt | SymNode | int | str,
    ) -> bool:
        """Check if two symbolic expressions are mathematically equivalent.

        Args:
            expr1 (Union[SymInt, SymNode, int, str]): First expression.
            expr2 (Union[SymInt, SymNode, int, str]): Second expression.

        Returns:
            bool: True if expressions evaluate to identical canonical representations.
        """
        if isinstance(expr1, int) and isinstance(expr2, int):
            return expr1 == expr2

        n1 = SymNode.to_node(expr1)
        n2 = SymNode.to_node(expr2)

        if str(n1) == str(n2):
            return True

        p1 = n1.to_polynomial()
        p2 = n2.to_polynomial()
        if p1 is not None and p2 is not None:
            return p1 == p2

        s1 = n1.simplify()
        s2 = n2.simplify()
        return str(s1) == str(s2)


class SymbolicConstraintTracker:
    """Disjoint-set constraint tracker to record dimension equalities across graphs."""

    def __init__(self) -> None:
        """Initialize SymbolicConstraintTracker."""
        self._parent: dict[str, str] = {}
        self._const_map: dict[str, int] = {}
        self._contradiction: bool = False
        self._error_msg: str = ""

    def _canonical_key(self, dim: int | str | SymNode | SymInt) -> str:
        """Return string representation key for dimension.

        Args:
            dim (Union[int, str, SymNode, SymInt]): Dimension to stringify.

        Returns:
            str: Canonical key string.
        """
        if isinstance(dim, SymInt):
            return str(dim.canonical())
        if isinstance(dim, SymNode):
            return str(dim.canonical())
        return str(dim)

    def _find(self, key: str) -> str:
        """Find representative with path compression.

        Args:
            key (str): Dimension key.

        Returns:
            str: Representative key.
        """
        if key not in self._parent:
            self._parent[key] = key
            return key
        if self._parent[key] != key:
            self._parent[key] = self._find(self._parent[key])
        return self._parent[key]

    def _resolve_const_dim(
        self,
        dim: int | str | SymNode | SymInt,
        key: str,
        root: str,
    ) -> int | None:
        """Resolve a dimension or its equivalence class root to a known constant integer.

        Args:
            dim (Union[int, str, SymNode, SymInt]): Dimension expression.
            key (str): Canonical key string for the dimension.
            root (str): Disjoint-set representative key.

        Returns:
            Optional[int]: Resolved constant integer if known, else None.
        """
        if isinstance(dim, int):
            return dim
        if isinstance(dim, SymConst):
            return dim.value
        if key.isdigit():
            return int(key)
        return self._const_map.get(root)

    def record_equality(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> None:
        """Record an equality constraint between two dimensions.

        Args:
            a (Union[int, str, SymNode, SymInt]): First dimension.
            b (Union[int, str, SymNode, SymInt]): Second dimension.

        Raises:
            ShapeMismatchError: If the recorded equality causes a contradiction.
        """
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                self._contradiction = True
                self._error_msg = f"Contradictory dimension constraints: {a} != {b}"
                raise ShapeMismatchError(self._error_msg)
            return

        key_a = self._canonical_key(a)
        key_b = self._canonical_key(b)

        root_a = self._find(key_a)
        root_b = self._find(key_b)

        const_a = self._resolve_const_dim(a, key_a, root_a)
        const_b = self._resolve_const_dim(b, key_b, root_b)

        if const_a is not None and const_b is not None and const_a != const_b:
            self._contradiction = True
            self._error_msg = f"Contradictory dimension constraints: {a} == {b} ({const_a} != {const_b})"
            raise ShapeMismatchError(self._error_msg)

        if root_a != root_b:
            self._parent[root_b] = root_a
        chosen_const = const_a if const_a is not None else const_b
        if chosen_const is not None:
            self._const_map[root_a] = chosen_const

    def unify(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> int | str | SymNode | SymInt:
        """Unify two dimensions according to broadcasting semantics.

        Args:
            a (Union[int, str, SymNode, SymInt]): First dimension.
            b (Union[int, str, SymNode, SymInt]): Second dimension.

        Returns:
            Union[int, str, SymNode, SymInt]: Unified dimension representative.

        Raises:
            ShapeMismatchError: If the dimensions cannot be unified.
        """
        if (isinstance(a, int) and a == 1) or (isinstance(a, SymConst) and a.value == 1) or str(a) == "1":
            return b
        if (isinstance(b, int) and b == 1) or (isinstance(b, SymConst) and b.value == 1) or str(b) == "1":
            return a

        self.record_equality(a, b)

        root = self._find(self._canonical_key(a))
        if root in self._const_map:
            val = self._const_map[root]
            if isinstance(a, SymInt) or isinstance(b, SymInt):
                return SymInt(val)
            if isinstance(a, SymNode) or isinstance(b, SymNode):
                return SymConst(val)
            return val

        return a

    def add_constraint(self, dim: str | SymNode | SymInt, val: int) -> None:
        """Assign a concrete integer value constraint to a symbolic dimension.

        Args:
            dim (Union[str, SymNode, SymInt]): Symbolic dimension.
            val (int): Concrete integer value.
        """
        self.record_equality(dim, val)

    def is_consistent(self) -> bool:
        """Check if all recorded constraints are mutually consistent.

        Returns:
            bool: True if no contradictions have been encountered.
        """
        return not self._contradiction

    def solve(self) -> dict[str, int]:
        """Solve for all bound variables and return their resolved values.

        Returns:
            dict[str, int]: Mapping of variable names to integer values.
        """
        solution: dict[str, int] = {}
        for var, root in list(self._parent.items()):
            actual_root = self._find(root)
            if actual_root in self._const_map:
                solution[var] = self._const_map[actual_root]
        return solution

    def substitute(
        self,
        dim_or_shape: int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...],
    ) -> int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...]:
        """Substitute known resolved values into dimension or shape tuple.

        Args:
            dim_or_shape (Union[int, str, SymNode, SymInt, tuple[...]]): Input dimension or shape.

        Returns:
            Union[int, str, SymNode, SymInt, tuple[...]]: Substituted dimension or shape.
        """
        if isinstance(dim_or_shape, tuple):
            return tuple(cast("int | str | SymNode | SymInt", self.substitute(d)) for d in dim_or_shape)

        key = self._canonical_key(dim_or_shape)
        root = self._find(key)
        if root in self._const_map:
            val = self._const_map[root]
            if isinstance(dim_or_shape, SymInt):
                return SymInt(val)
            if isinstance(dim_or_shape, SymNode):
                return SymConst(val)
            if isinstance(dim_or_shape, int):
                return val
            return val

        return dim_or_shape

    def track_graph(self, graph: LogicalGraph) -> None:
        """Traverse a logical graph to extract and unify dimension equality constraints.

        Args:
            graph (LogicalGraph): LogicalGraph or IRGraph instance.
        """
        nodes = getattr(graph, "nodes", [])
        for node in nodes:
            inputs = getattr(node, "inputs", [])
            output_shape = getattr(node, "shape", None)
            op = getattr(node, "op", "")
            if op == "matmul" and len(inputs) == BINARY_OP_ARITY:
                s1 = getattr(inputs[0], "shape", None)
                s2 = getattr(inputs[1], "shape", None)
                if s1 and s2 and len(s1) >= 1 and len(s2) >= 1:
                    self.record_equality(s1[-1], s2[-2 if len(s2) > 1 else -1])
            elif op in ("add", "sub", "mul", "div") and len(inputs) == BINARY_OP_ARITY:
                s1 = getattr(inputs[0], "shape", None)
                s2 = getattr(inputs[1], "shape", None)
                if s1 and s2:
                    broadcast_shapes(s1, s2, tracker=self)
            if output_shape is not None and inputs:
                pass


def _to_str_shape(shape: tuple[int | str | SymNode | SymInt, ...]) -> tuple[int | str, ...]:
    """Convert shape to string representation.

    Args:
        shape (tuple[Union[int, str, SymNode, SymInt], ...]): The shape tuple.

    Returns:
        tuple[Union[int, str], ...]: The string represented shape.
    """
    return tuple(str(x) if isinstance(x, (SymInt, SymNode)) else x for x in shape)


def _from_str_shape(shape: tuple[int | str, ...]) -> tuple[int | SymInt, ...]:
    """Convert shape from string representation.

    Args:
        shape (tuple[Union[int, str], ...]): The string represented shape tuple.

    Returns:
        tuple[Union[int, SymInt], ...]: The integer or SymInt shape.
    """
    out_shape: list[int | SymInt] = []
    for dim in shape:
        if isinstance(dim, str) and not dim.isdigit():
            out_shape.append(SymInt(dim))
        else:
            out_shape.append(int(dim))
    return tuple(out_shape)


class ShapeTracker:
    """Calculate exact output shapes given input TensorSpecs."""

    @staticmethod
    def infer_elementwise(inputs: list[TensorSpec]) -> tuple[int | SymInt, ...]:
        """Infer shape for elementwise operations requiring broadcasting.

        Args:
            inputs (list[TensorSpec]): The inputs for the operation.

        Returns:
            tuple[Union[int, SymInt], ...]: The inferred shape or computed result.
        """
        if not inputs:
            return ()

        shape = cast("Callable[..., tuple[Union[int, str], ...]]", _to_str_shape)(inputs[0].shape)
        for i in range(1, len(inputs)):
            shape = cast("Callable[..., tuple[Union[int, str], ...]]", _to_str_shape)(broadcast_shapes(shape, cast("Callable[..., tuple[Union[int, str], ...]]", _to_str_shape)(inputs[i].shape)))

        return _from_str_shape(shape)

    @staticmethod
    def infer_matmul(
        input1: TensorSpec,
        input2: TensorSpec,
    ) -> tuple[int | SymInt, ...]:
        """Infer shape for matrix multiplication.

        Args:
            input1 (TensorSpec): The first input spec.
            input2 (TensorSpec): The second input spec.

        Returns:
            tuple[Union[int, SymInt], ...]: The inferred shape or computed result.
        """
        s1 = cast("Callable[..., tuple[Union[int, str], ...]]", _to_str_shape)(input1.shape)
        s2 = cast("Callable[..., tuple[Union[int, str], ...]]", _to_str_shape)(input2.shape)

        out_shape_str = matmul_shape(s1, s2)

        return cast("Callable[..., tuple[Union[int, SymInt], ...]]", _from_str_shape)(out_shape_str)

    @staticmethod
    def update_from_feedback(
        feedback_telemetry: dict[str, list[int] | tuple[int, ...] | dict[str, list[int]]],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Update shape tracking state and unify constraints from runtime telemetry feedback.

        Args:
            feedback_telemetry (dict[str, list[int] | tuple[int, ...] | dict[str, list[int]]]):
                Telemetry payload mapping node IDs to concrete observed shapes or containing a 'shapes' mapping.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker to unify.

        Returns:
            dict[str, tuple[int, ...]]: Dictionary of resolved concrete shapes per node.
        """
        raw_shapes: dict[str, list[int] | tuple[int, ...]] = {}
        if "shapes" in feedback_telemetry and isinstance(feedback_telemetry["shapes"], dict):
            raw_shapes = dict(feedback_telemetry["shapes"])
        else:
            for k, v in feedback_telemetry.items():
                if isinstance(v, (list, tuple)):
                    raw_shapes[k] = v

        resolved: dict[str, tuple[int, ...]] = {}
        for node_id, shape_val in raw_shapes.items():
            concrete_tuple = tuple(int(dim) for dim in shape_val)
            resolved[str(node_id)] = concrete_tuple
            if tracker is not None:
                for dim in concrete_tuple:
                    tracker.record_equality(dim, dim)

        return resolved

    @staticmethod
    def _record_symbolic_dim_bound(
        dim: int | str | SymNode | SymInt,
        concrete_val: int,
        resolved_bounds: dict[str, int],
        tracker: SymbolicConstraintTracker | None,
    ) -> None:
        """Record a resolved concrete bound for a single symbolic dimension if applicable.

        Args:
            dim (Union[int, str, SymNode, SymInt]): Symbolic or static dimension entry.
            concrete_val (int): Concrete integer bound from telemetry.
            resolved_bounds (dict[str, int]): Mapping of symbolic names to resolved bounds to mutate.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.

        Returns:
            None: Updates resolved_bounds and tracker in-place.
        """
        var_name: str | None = None
        if isinstance(dim, SymInt):
            var_name = dim.name
        elif isinstance(dim, SymNode):
            var_name = str(dim)
        elif isinstance(dim, str) and not dim.isdigit():
            var_name = dim

        if var_name is not None:
            resolved_bounds[var_name] = concrete_val
            if tracker is not None:
                tracker.record_equality(dim, concrete_val)

    @staticmethod
    def resolve_dynamic_bounds(
        graph: LogicalGraph,
        feedback_telemetry: dict[str, list[int] | tuple[int, ...] | dict[str, list[int]]],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, int]:
        """Resolve dynamic SymInt variables across an IR graph into static integer bounds using telemetry.

        Args:
            graph (LogicalGraph): LogicalGraph or IRGraph instance containing nodes with symbolic dimensions.
            feedback_telemetry (dict[str, list[int] | tuple[int, ...] | dict[str, list[int]]]):
                In-browser execution telemetry with observed concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.

        Returns:
            dict[str, int]: Mapping of symbolic dimension names to their resolved static integer bounds.
        """
        concrete_shapes = ShapeTracker.update_from_feedback(feedback_telemetry, tracker)
        resolved_bounds: dict[str, int] = {}

        raw_nodes = getattr(graph, "nodes", {})
        node_iterable = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes

        for node in node_iterable:
            node_id = str(getattr(node, "id", ""))
            if node_id not in concrete_shapes:
                continue

            observed_shape = concrete_shapes[node_id]
            shape_meta = getattr(node, "shape_metadata", None)
            if shape_meta is None:
                shape_meta = getattr(node, "shape", None)

            if isinstance(shape_meta, (tuple, list)):
                new_shape: list[int] = []
                for idx, dim in enumerate(shape_meta):
                    concrete_val = observed_shape[idx] if idx < len(observed_shape) else 1
                    ShapeTracker._record_symbolic_dim_bound(dim, concrete_val, resolved_bounds, tracker)
                    new_shape.append(concrete_val)

                if hasattr(node, "shape_metadata"):
                    node.shape_metadata = tuple(new_shape)
                if hasattr(node, "shape"):
                    node.shape = tuple(new_shape)

        return resolved_bounds


ShapeType = tuple[Union[int, str, SymNode, SymInt], ...]


def _broadcast_dim(
    a: int | str | SymNode | SymInt,
    b: int | str | SymNode | SymInt,
    tracker: SymbolicConstraintTracker | None = None,
) -> int | str | SymNode | SymInt:
    """Broadcast two dimensions under constraint unification.

    Args:
        a (Union[int, str, SymNode, SymInt]): The first dimension.
        b (Union[int, str, SymNode, SymInt]): The second dimension.
        tracker (Optional[SymbolicConstraintTracker]): Optional constraint tracker.

    Returns:
        Union[int, str, SymNode, SymInt]: The broadcasted dimension.

    Raises:
        ShapeMismatchError: If the dimensions are incompatible.
    """
    if a == b:
        return a

    if isinstance(a, SymConst):
        a = a.value
    if isinstance(b, SymConst):
        b = b.value

    if a == 1 or (isinstance(a, str) and a == "1"):
        return b
    if b == 1 or (isinstance(b, str) and b == "1"):
        return a

    if SymbolicSolver.is_consistent(a, b):
        return a

    if isinstance(a, int) and isinstance(b, int):
        msg = f"Incompatible dimensions for broadcasting: {a} and {b}"
        raise ShapeMismatchError(msg)

    if tracker is not None:
        return tracker.unify(a, b)

    if (isinstance(a, (str, SymNode, SymInt))) and (isinstance(b, (str, SymNode, SymInt))):
        return a

    msg = f"Incompatible dimensions for broadcasting: {a} and {b}"
    raise ShapeMismatchError(msg)


def broadcast_shapes(
    shape_a: ShapeType,
    shape_b: ShapeType,
    tracker: SymbolicConstraintTracker | None = None,
) -> ShapeType:
    """Broadcast two shapes together according to numpy rules with constraint unification.

    Args:
        shape_a (ShapeType): First shape.
        shape_b (ShapeType): Second shape.
        tracker (Optional[SymbolicConstraintTracker]): Optional constraint tracker.

    Returns:
        ShapeType: The broadcasted shape.
    """
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + shape_a
    pad_b = (1,) * (max_len - len(shape_b)) + shape_b
    return tuple(_broadcast_dim(a, b, tracker=tracker) for a, b in zip(pad_a, pad_b))


def _matmul_shape_1d(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a 1D dot product.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ValueError: If the shapes are incompatible.
    """
    if not SymbolicSolver.is_consistent(shape_a[0], shape_b[0]):
        msg = f"Incompatible 1D dot product shapes: {shape_a}, {shape_b}"
        raise ValueError(msg)
    return ()


def _matmul_shape_2d(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a 2D matrix multiplication.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If the shapes are incompatible.
    """
    if not SymbolicSolver.is_consistent(shape_a[1], shape_b[0]):
        msg = f"Incompatible 2D matmul shapes: {shape_a}, {shape_b}"
        raise ShapeMismatchError(msg)
    return (shape_a[0], shape_b[1])


def _get_matmul_dims(
    shape_a: ShapeType, shape_b: ShapeType
) -> tuple[
    int | str | SymNode | SymInt,
    int | str | SymNode | SymInt,
    int | str | SymNode | SymInt,
    int | str | SymNode | SymInt,
]:
    """Extract matrix multiplication dimensions (m, k_a, k_b, n).

    Args:
        shape_a (ShapeType): The shape_a parameter.
        shape_b (ShapeType): The shape_b parameter.

    Returns:
        tuple[Union[int, str, SymNode, SymInt], ...]: (m_dim, k_dim_a, k_dim_b, n_dim).
    """
    m_dim = shape_a[-2] if len(shape_a) > 1 else 1
    k_dim_a = shape_a[-1]
    k_dim_b = shape_b[-2] if len(shape_b) > 1 else shape_b[-1]
    n_dim = shape_b[-1] if len(shape_b) > 1 else 1
    return m_dim, k_dim_a, k_dim_b, n_dim


def _get_batch_dims(shape: ShapeType) -> ShapeType:
    """Extract batch dimensions from shape.

    Args:
        shape (ShapeType): The shape parameter.

    Returns:
        ShapeType: Batch dimensions.
    """
    return shape[:-2] if len(shape) > MATRIX_RANK else ()


def _matmul_shape_batched(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a batched matrix multiplication.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If the inner dimensions are incompatible.
    """
    batch_a = _get_batch_dims(shape_a)
    batch_b = _get_batch_dims(shape_b)
    out_batch = broadcast_shapes(batch_a, batch_b)

    m_dim, k_dim_a, k_dim_b, n_dim = _get_matmul_dims(shape_a, shape_b)

    if not SymbolicSolver.is_consistent(k_dim_a, k_dim_b):
        msg = f"Incompatible inner dimensions for matmul: {k_dim_a} and {k_dim_b}"
        raise ShapeMismatchError(msg)

    out_shape = list(out_batch)
    if len(shape_a) > 1:
        out_shape.append(m_dim)
    if len(shape_b) > 1:
        out_shape.append(n_dim)

    return tuple(out_shape)


def matmul_shape(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a matrix multiplication.

    Args:
        shape_a (ShapeType): First shape.
        shape_b (ShapeType): Second shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If shapes are incompatible or scalars.
    """
    if len(shape_a) == 0 or len(shape_b) == 0:
        msg = "Scalars cannot be matrix multiplied."
        raise ShapeMismatchError(msg)

    if len(shape_a) == 1 and len(shape_b) == 1:
        return _matmul_shape_1d(shape_a, shape_b)

    if len(shape_a) == MATRIX_RANK and len(shape_b) == MATRIX_RANK:
        return _matmul_shape_2d(shape_a, shape_b)

    return _matmul_shape_batched(shape_a, shape_b)


def _normalize_single_axis(axis: int, ndim: int) -> int:
    """Normalize a single negative axis to be positive.

    Args:
        axis (int): The axis.
        ndim (int): Number of dimensions.

    Returns:
        int: Normalized axis.

    Raises:
        ShapeMismatchError: If axis is out of bounds.
    """
    if axis < -ndim or axis >= ndim:
        msg = f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
        raise ShapeMismatchError(msg)
    if axis < 0:
        return axis + ndim
    return axis


def normalize_axis(
    axis: int | tuple[int, ...] | list[int],
    ndim: int,
) -> int | tuple[int, ...]:
    """Normalize a negative axis or tuple of axes to be positive.

    Args:
        axis (Union[int, tuple[int, ...], list[int]]): The axis to operate along.
        ndim (int): Number of dimensions.

    Returns:
        Union[int, tuple[int, ...]]: Normalized axis or axes.

    Raises:
        TypeError: If axis has an invalid type.
    """
    if isinstance(axis, int):
        return _normalize_single_axis(axis, ndim)
    if isinstance(axis, (tuple, list)):
        return tuple(_normalize_single_axis(ax, ndim) for ax in axis)
    msg = f"Invalid type for axis: {type(axis)}"
    raise TypeError(msg)
