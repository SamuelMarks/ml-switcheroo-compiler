"""Type & Shape System for IR with symbolic term rewriting and constraint tracking."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import os
from typing import TYPE_CHECKING, Union

import yaml
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.errors import ShapeMismatchError

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import TensorSpec

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
            c = other.get_const()
            if c == 0:
                return None
            new_terms: dict[MonomialType, int] = {}
            for mono, coeff in self.terms.items():
                if coeff % c != 0:
                    return None
                new_terms[mono] = coeff // c
            return Polynomial(new_terms)

        if len(other.terms) == 1:
            div_mono, div_coeff = next(iter(other.terms.items()))
            new_terms = {}
            for mono, coeff in self.terms.items():
                if coeff % div_coeff != 0:
                    return None
                mono_dict = dict(mono)
                div_dict = dict(div_mono)
                for var, power in div_dict.items():
                    if mono_dict.get(var, 0) < power:
                        return None
                    mono_dict[var] -= power
                    if mono_dict[var] == 0:
                        del mono_dict[var]
                quot_mono = tuple(sorted(mono_dict.items()))
                new_terms[quot_mono] = coeff // div_coeff
            return Polynomial(new_terms)

        return None

    def canonical_str(self) -> str:
        """Return canonical deterministic string representation of polynomial.

        Returns:
            str: Deterministic string format with terms sorted consistently.
        """
        if self.is_zero():
            return "0"

        def sort_key(mono: MonomialType) -> tuple[int, list[str]]:
            """Sort key for monomial terms by descending degree and variable names.

            Args:
                mono (MonomialType): Monomial representation.

            Returns:
                tuple[int, list[str]]: Comparison sort key.
            """
            degree = sum(p for _, p in mono)
            vars_sorted = [v for v, _ in mono]
            return (-degree, vars_sorted)

        sorted_monos = sorted(self.terms.keys(), key=sort_key)
        parts: list[str] = []
        for i, mono in enumerate(sorted_monos):
            coeff = self.terms[mono]
            sign = ""
            if i > 0:
                if coeff > 0:
                    sign = " + "
                else:
                    sign = " - "
                    coeff = abs(coeff)
            elif coeff < 0:
                sign = "-"
                coeff = abs(coeff)

            if len(mono) == 0:
                part = f"{coeff}"
            else:
                var_factors = []
                for var, pow_val in mono:
                    if pow_val == 1:
                        var_factors.append(var)
                    else:
                        var_factors.append(f"{var}^{pow_val}")
                var_str = "*".join(var_factors)
                if coeff == 1:
                    part = var_str
                else:
                    part = f"{coeff}*{var_str}"
            parts.append(f"{sign}{part}")

        return "".join(parts)

    def __eq__(self, other: object) -> bool:
        """Evaluate equality between two polynomials.

        Args:
            other (object): Other object to compare against.

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
        """
        raise NotImplementedError

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

    def __radd__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-add two symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Left operand.

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

    def __rmul__(self, other: int | str | SymNode | SymInt) -> SymBinaryOp:
        """Right-multiply symbolic expressions.

        Args:
            other (Union[int, str, SymNode, SymInt]): Left operand.

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
            other (object): Other object.

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
            other (object): Other object.

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

    def simplify(self) -> SymNode:
        """Apply algebraic rewrite rules to simplify binary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        if isinstance(s_left, SymConst) and isinstance(s_right, SymConst):
            v1, v2 = s_left.value, s_right.value
            if self.op == "+":
                return SymConst(v1 + v2)
            if self.op == "-":
                return SymConst(v1 - v2)
            if self.op == "*":
                return SymConst(v1 * v2)
            if self.op == "//" and v2 != 0:
                return SymConst(v1 // v2)

        if self.op == "+":
            if isinstance(s_right, SymConst) and s_right.value == 0:
                return s_left
            if isinstance(s_left, SymConst) and s_left.value == 0:
                return s_right

        elif self.op == "-":
            if isinstance(s_right, SymConst) and s_right.value == 0:
                return s_left
            if s_left == s_right:
                return SymConst(0)
            if isinstance(s_left, SymBinaryOp) and s_left.op == "+":
                if s_left.right == s_right:
                    return s_left.left
                if s_left.left == s_right:
                    return s_left.right

        elif self.op == "*":
            if isinstance(s_right, SymConst) and s_right.value == 1:
                return s_left
            if isinstance(s_left, SymConst) and s_left.value == 1:
                return s_right
            if (isinstance(s_right, SymConst) and s_right.value == 0) or (isinstance(s_left, SymConst) and s_left.value == 0):
                return SymConst(0)

        elif self.op == "//":
            if isinstance(s_right, SymConst) and s_right.value == 1:
                return s_left
            if isinstance(s_left, SymConst) and s_left.value == 0:
                return SymConst(0)
            if s_left == s_right and not (isinstance(s_right, SymConst) and s_right.value == 0):
                return SymConst(1)
            if isinstance(s_left, SymBinaryOp) and s_left.op == "*":
                if s_left.right == s_right:
                    return s_left.left
                if s_left.left == s_right:
                    return s_left.right

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
            ZeroDivisionError: If dividing by zero.
            ValueError: If operator is unsupported.
        """
        v1 = self.left.eval(env)
        v2 = self.right.eval(env)
        if self.op == "+":
            return v1 + v2
        if self.op == "-":
            return v1 - v2
        if self.op == "*":
            return v1 * v2
        if self.op == "//":
            if v2 == 0:
                msg = "Division by zero in symbolic evaluation."
                raise ZeroDivisionError(msg)
            return v1 // v2
        msg = f"Unsupported operator '{self.op}'."
        raise ValueError(msg)

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
        """Check structural equality.

        Args:
            other (object): Other object.

        Returns:
            bool: True if op, left, and right match.
        """
        if isinstance(other, SymBinaryOp):
            return self.op == other.op and self.left == other.left and self.right == other.right
        return False

    def __hash__(self) -> int:
        """Compute hash code.

        Returns:
            int: Hash integer.
        """
        return hash((self.op, self.left, self.right))


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

    def sort_key(mono: MonomialType) -> tuple[int, list[str]]:
        """Sort key for monomial terms by descending degree and variable names.

        Args:
            mono (MonomialType): Monomial representation.

        Returns:
            tuple[int, list[str]]: Comparison sort key.
        """
        degree = sum(p for _, p in mono)
        vars_sorted = [v for v, _ in mono]
        return (-degree, vars_sorted)

    sorted_monos = sorted(poly.terms.keys(), key=sort_key)
    res_node: SymNode | None = None

    for mono in sorted_monos:
        coeff = poly.terms[mono]
        is_sub = False
        if res_node is not None and coeff < 0:
            is_sub = True
            coeff = abs(coeff)

        term_node: SymNode | None = None
        if coeff != 1 or len(mono) == 0:
            term_node = SymConst(coeff)

        for var, power in mono:
            for _ in range(power):
                if term_node is None:
                    term_node = SymVar(var)
                else:
                    term_node = SymBinaryOp("*", term_node, SymVar(var))

        if term_node is None:
            term_node = SymConst(0)

        if res_node is None:
            if coeff < 0:
                res_node = SymBinaryOp("-", SymConst(0), term_node)
            else:
                res_node = term_node
        elif is_sub:
            res_node = SymBinaryOp("-", res_node, term_node)
        else:
            res_node = SymBinaryOp("+", res_node, term_node)

    return res_node or SymConst(0)


def _tokenize_expr(expr_str: str) -> list[str]:
    """Tokenize a symbolic expression string into component tokens.

    Args:
        expr_str (str): Input expression string.

    Returns:
        list[str]: Sequence of string tokens.
    """
    tokens: list[str] = []
    i = 0
    s = expr_str.strip()
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()+*":
            tokens.append(c)
            i += 1
        elif c == "-":
            tokens.append(c)
            i += 1
        elif c == "/":
            if i + 1 < n and s[i + 1] == "/":
                tokens.append("//")
                i += 2
            else:
                tokens.append("//")
                i += 1
        elif c.isdigit():
            start = i
            while i < n and s[i].isdigit():
                i += 1
            tokens.append(s[start:i])
        elif c.isalpha() or c == "_":
            start = i
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(s[start:i])
        else:
            i += 1
    return tokens


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

    idx = 0

    def parse_expr() -> SymNode:
        """Parse additive binary expressions.

        Returns:
            SymNode: Root parsed expression node.
        """
        nonlocal idx
        node = parse_term()
        while idx < len(tokens) and tokens[idx] in ("+", "-"):
            op = tokens[idx]
            idx += 1
            right = parse_term()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_term() -> SymNode:
        """Parse multiplicative binary expressions.

        Returns:
            SymNode: Root parsed term node.
        """
        nonlocal idx
        node = parse_factor()
        while idx < len(tokens) and tokens[idx] in ("*", "//"):
            op = tokens[idx]
            idx += 1
            right = parse_factor()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_factor() -> SymNode:
        """Parse factor, parenthesized expression, or unary negative.

        Returns:
            SymNode: Root parsed factor node.
        """
        nonlocal idx
        if idx >= len(tokens):
            return SymConst(0)
        tok = tokens[idx]
        if tok == "(":
            idx += 1
            node = parse_expr()
            if idx < len(tokens) and tokens[idx] == ")":
                idx += 1
            return node
        if tok == "-":
            idx += 1
            inner = parse_factor()
            return SymBinaryOp("-", SymConst(0), inner)
        idx += 1
        if tok.isdigit():
            return SymConst(int(tok))
        return SymVar(tok)

    try:
        node = parse_expr()
        if idx == len(tokens):
            return node
    except (ValueError, IndexError):
        pass

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
        """Evaluate __add__ operation.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", self.node, other_node))

    def __radd__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate __radd__ operation.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", other_node, self.node))

    def __sub__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate __sub__ operation.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", self.node, other_node))

    def __rsub__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate __rsub__ operation.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", other_node, self.node))

    def __mul__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate __mul__ operation.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", self.node, other_node))

    def __rmul__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate __rmul__ operation.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", other_node, self.node))

    def __floordiv__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate __floordiv__ operation.

        Args:
            other (Union[SymInt, int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", self.node, other_node))

    def __rfloordiv__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate __rfloordiv__ operation.

        Args:
            other (Union[int, SymNode, str]): The other operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", other_node, self.node))

    def __str__(self) -> str:
        """Evaluate __str__ operation.

        Returns:
            str: String expression.
        """
        return str(self.node)

    def __repr__(self) -> str:
        """Evaluate __repr__ operation.

        Returns:
            str: Debugging representation.
        """
        return f"SymInt({self.node})"

    def __hash__(self) -> int:
        """Evaluate __hash__ operation.

        Returns:
            int: Hash integer.
        """
        return hash(self.expr)

    def __eq__(self, other: object) -> bool:
        """Evaluate __eq__ operation using canonical polynomial consistency.

        Args:
            other (object): The other object to compare.

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

        const_a: int | None = None
        const_b: int | None = None

        if isinstance(a, int):
            const_a = a
        elif isinstance(a, SymConst):
            const_a = a.value
        elif key_a.isdigit():
            const_a = int(key_a)
        elif root_a in self._const_map:
            const_a = self._const_map[root_a]

        if isinstance(b, int):
            const_b = b
        elif isinstance(b, SymConst):
            const_b = b.value
        elif key_b.isdigit():
            const_b = int(key_b)
        elif root_b in self._const_map:
            const_b = self._const_map[root_b]

        if const_a is not None and const_b is not None and const_a != const_b:
            self._contradiction = True
            self._error_msg = f"Contradictory dimension constraints: {a} == {b} ({const_a} != {const_b})"
            raise ShapeMismatchError(self._error_msg)

        if root_a != root_b:
            self._parent[root_b] = root_a
            chosen_const = const_a if const_a is not None else const_b
            if chosen_const is not None:
                self._const_map[root_a] = chosen_const
        elif const_a is not None:
            self._const_map[root_a] = const_a

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
            return tuple(self.substitute(d) for d in dim_or_shape)

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

    def track_graph(self, graph: object) -> None:
        """Traverse a logical graph to extract and unify dimension equality constraints.

        Args:
            graph (object): LogicalGraph or IRGraph instance.
        """
        nodes = getattr(graph, "nodes", [])
        for node in nodes:
            inputs = getattr(node, "inputs", [])
            output_shape = getattr(node, "shape", None)
            op = getattr(node, "op", "")
            if op == "matmul" and len(inputs) == MAGIC_VAL_2:
                s1 = getattr(inputs[0], "shape", None)
                s2 = getattr(inputs[1], "shape", None)
                if s1 and s2 and len(s1) >= 1 and len(s2) >= 1:
                    self.record_equality(s1[-1], s2[-2 if len(s2) > 1 else -1])
            elif op in ("add", "sub", "mul", "div") and len(inputs) == MAGIC_VAL_2:
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

        shape = _to_str_shape(inputs[0].shape)
        for i in range(1, len(inputs)):
            shape = broadcast_shapes(shape, _to_str_shape(inputs[i].shape))

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
        s1 = _to_str_shape(input1.shape)
        s2 = _to_str_shape(input2.shape)

        out_shape_str = matmul_shape(s1, s2)

        return _from_str_shape(out_shape_str)

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
            raw_shapes = feedback_telemetry["shapes"]
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
    def resolve_dynamic_bounds(
        graph: object,
        feedback_telemetry: dict[str, list[int] | tuple[int, ...] | dict[str, list[int]]],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, int]:
        """Resolve dynamic SymInt variables across an IR graph into static integer bounds using telemetry.

        Args:
            graph (object): LogicalGraph or IRGraph instance containing nodes with symbolic dimensions.
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
                    if isinstance(dim, SymInt):
                        var_name = dim.name
                        resolved_bounds[var_name] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
                    elif isinstance(dim, SymNode):
                        var_name = str(dim)
                        resolved_bounds[var_name] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
                    elif isinstance(dim, str) and not dim.isdigit():
                        resolved_bounds[dim] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
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
    return shape[:-2] if len(shape) > MAGIC_VAL_2 else ()


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

    if len(shape_a) == MAGIC_VAL_2 and len(shape_b) == MAGIC_VAL_2:
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
