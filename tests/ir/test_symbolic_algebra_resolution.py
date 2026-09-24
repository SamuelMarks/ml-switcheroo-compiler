"""Exhaustive tests for non-polynomial symbolic algebra, simplification, and piecewise evaluation."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.ir.shape_system import (
    SymBinaryOp,
    SymConst,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)


def test_non_polynomial_operators_evaluation() -> None:
    """Test evaluation of floor division, modulo, power, min, max, and comparison operators."""
    x = SymVar("x")
    y = SymVar("y")
    env = {"x": 17, "y": 5}

    # Floor division
    div_expr = x // y
    assert div_expr.eval(env) == 3

    # Negative floor division: -17 // 5 == -4
    neg_div = (-x) // y
    assert neg_div.eval(env) == -4

    # Modulo: 17 % 5 == 2
    mod_expr = x % y
    assert mod_expr.eval(env) == 2

    # Power: 2 ** 3 == 8
    pow_expr = SymNode.to_node(2) ** SymNode.to_node(3)
    assert pow_expr.eval({}) == 8

    # Min and Max
    min_expr = SymBinaryOp("min", x, y)
    max_expr = SymBinaryOp("max", x, y)
    assert min_expr.eval(env) == 5
    assert max_expr.eval(env) == 17

    # Relational comparisons
    assert SymBinaryOp("==", x, y).eval(env) == 0
    assert SymBinaryOp("!=", x, y).eval(env) == 1
    assert SymBinaryOp(">", x, y).eval(env) == 1
    assert SymBinaryOp("<", x, y).eval(env) == 0
    assert SymBinaryOp(">=", x, y).eval(env) == 1
    assert SymBinaryOp("<=", x, y).eval(env) == 0

    # Division and modulo by zero detection
    zero_env = {"x": 10, "y": 0}
    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        div_expr.eval(zero_env)

    with pytest.raises(ZeroDivisionError, match="Modulo by zero"):
        mod_expr.eval(zero_env)


def test_symbolic_simplification_rules() -> None:
    """Test algebraic simplification rules for floor division, modulo, min, and max."""
    n = SymVar("n")
    k = SymConst(4)

    # (k * n) // k == n
    mul_expr = k * n
    div_mul = mul_expr // k
    simplified = div_mul.simplify()
    assert simplified == n

    # (n * k) // k == n
    mul_rev = n * k
    assert (mul_rev // k).simplify() == n

    # 0 // n == 0
    assert (SymConst(0) // n).simplify() == SymConst(0)

    # n // 1 == n
    assert (n // 1).simplify() == n

    # n // n == 1
    assert (n // n).simplify() == SymConst(1)

    # Modulo simplifications: n % 1 == 0, n % n == 0, 0 % n == 0
    assert (n % 1).simplify() == SymConst(0)
    assert (n % n).simplify() == SymConst(0)
    assert (SymConst(0) % n).simplify() == SymConst(0)

    # Idempotence: min(n, n) == n, max(n, n) == n
    assert SymBinaryOp("min", n, n).simplify() == n
    assert SymBinaryOp("max", n, n).simplify() == n

    # Constant folding for min and max
    assert SymBinaryOp("min", SymConst(3), SymConst(7)).simplify() == SymConst(3)
    assert SymBinaryOp("max", SymConst(3), SymConst(7)).simplify() == SymConst(7)


def test_unary_operations_and_piecewise() -> None:
    """Test SymUnaryOp and SymPiecewise conditional branching."""
    x = SymVar("x")

    # Unary operations: abs, ceil, floor, neg
    abs_node = abs(x)
    assert isinstance(abs_node, SymUnaryOp)
    assert abs_node.eval({"x": -42}) == 42
    assert abs_node.eval({"x": 42}) == 42

    neg_node = -x
    assert neg_node.eval({"x": 10}) == -10
    assert (-SymConst(5)).simplify() == SymConst(-5)
    assert abs(SymConst(-9)).simplify() == SymConst(9)

    ceil_node = SymUnaryOp("ceil", SymConst(7))
    assert ceil_node.eval({}) == 7
    assert ceil_node.simplify() == SymConst(7)

    floor_node = SymUnaryOp("floor", SymConst(7))
    assert floor_node.eval({}) == 7
    assert floor_node.simplify() == SymConst(7)

    assert str(neg_node) == "-(x)"
    assert str(abs_node) == "abs(x)"
    assert repr(abs_node) == "SymUnaryOp(abs, SymVar('x'))"
    assert abs_node.free_vars() == {"x"}

    with pytest.raises(ValueError, match="Unsupported unary operator"):
        SymUnaryOp("invalid_op", SymConst(1)).eval({})

    # SymPiecewise conditional nodes
    cond_1 = lambda env: env["x"] > 10
    cond_2 = lambda env: env["x"] > 0
    piecewise = SymPiecewise(
        cases=[
            (cond_1, SymConst(100)),
            (cond_2, SymConst(50)),
        ],
        default=SymConst(0),
    )

    assert piecewise.eval({"x": 20}) == 100
    assert piecewise.eval({"x": 5}) == 50
    assert piecewise.eval({"x": -5}) == 0

    # Piecewise with SymNode condition and boolean condition
    sym_cond = SymBinaryOp("==", x, SymConst(42))
    pw2 = SymPiecewise(cases=[(sym_cond, SymConst(999)), (True, SymConst(111))], default=SymConst(0))
    assert pw2.eval({"x": 42}) == 999
    assert pw2.eval({"x": 1}) == 111

    # Free vars union
    assert pw2.free_vars() == {"x"}
    assert piecewise.simplify() is not None
