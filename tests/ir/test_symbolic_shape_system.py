"""Unit tests for symbolic shape tracking, dynamic dimensions, and algebraic solver."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.ir.shape_system import (
    AlgebraicRuleSetModel,
    Polynomial,
    SymBinaryOp,
    SymbolicConstraintTracker,
    SymbolicSolver,
    SymConst,
    SymInt,
    SymNode,
    SymVar,
    _broadcast_dim,
    _parse_sym_str,
    _poly_to_symnode,
    _tokenize_expr,
    broadcast_shapes,
    load_algebraic_rules,
    matmul_shape,
    normalize_axis,
)


class MockLogicalNode:
    """Mock logical graph node for constraint tracking tests.

    Attributes:
        op (str): Operation name.
        inputs (list[MockLogicalNode]): Predecessor nodes.
        shape (tuple[Union[int, str, SymNode, SymInt], ...]): Output tensor shape.
    """

    def __init__(
        self,
        op: str,
        inputs: list[MockLogicalNode],
        shape: tuple[int | str | SymNode | SymInt, ...],
    ) -> None:
        """Initialize MockLogicalNode.

        Args:
            op (str): Operation name.
            inputs (list[MockLogicalNode]): Predecessor nodes.
            shape (tuple[Union[int, str, SymNode, SymInt], ...]): Output shape tuple.
        """
        self.op: str = op
        self.inputs: list[MockLogicalNode] = inputs
        self.shape: tuple[int | str | SymNode | SymInt, ...] = shape


class MockLogicalGraph:
    """Mock logical graph container for constraint tracking tests.

    Attributes:
        nodes (list[MockLogicalNode]): Graph nodes in topological order.
    """

    def __init__(self, nodes: list[MockLogicalNode]) -> None:
        """Initialize MockLogicalGraph.

        Args:
            nodes (list[MockLogicalNode]): List of nodes.
        """
        self.nodes: list[MockLogicalNode] = nodes


def test_algebraic_rules_loading() -> None:
    """Verify that declarative algebraic rules are correctly loaded from YAML."""
    rule_set = load_algebraic_rules()
    assert isinstance(rule_set, AlgebraicRuleSetModel)
    assert len(rule_set.rules) > 0

    rule_names = {r.name for r in rule_set.rules}
    assert "add_zero_right" in rule_names
    assert "add_zero_left" in rule_names
    assert "mul_one_right" in rule_names
    assert "mul_one_left" in rule_names
    assert "mul_zero_right" in rule_names
    assert "sub_self" in rule_names
    assert "add_sub_cancel" in rule_names
    assert "mul_div_cancel" in rule_names

    # Test loading from non-existent file falls back gracefully
    fallback = load_algebraic_rules("/nonexistent/path/rules.yaml")
    assert isinstance(fallback, AlgebraicRuleSetModel)
    assert len(fallback.rules) == 0


def test_symnode_algebraic_rewriting_identities() -> None:
    """Test algebraic rewrite patterns for identities: x + 0, 0 + x, x * 1, 1 * x."""
    x = SymVar("x")
    zero = SymConst(0)
    one = SymConst(1)

    # x + 0 -> x
    add_r0 = SymBinaryOp("+", x, zero).simplify()
    assert add_r0 == x

    # 0 + x -> x
    add_l0 = SymBinaryOp("+", zero, x).simplify()
    assert add_l0 == x

    # x * 1 -> x
    mul_r1 = SymBinaryOp("*", x, one).simplify()
    assert mul_r1 == x

    # 1 * x -> x
    mul_l1 = SymBinaryOp("*", one, x).simplify()
    assert mul_l1 == x


def test_symnode_algebraic_rewriting_cancellations() -> None:
    """Test algebraic rewrite patterns for cancellations: x * 0, x - x, (x + y) - y, (x * y) // y."""
    x = SymVar("x")
    y = SymVar("y")
    zero = SymConst(0)

    # x * 0 -> 0
    mul_r0 = SymBinaryOp("*", x, zero).simplify()
    assert mul_r0 == zero

    # 0 * x -> 0
    mul_l0 = SymBinaryOp("*", zero, x).simplify()
    assert mul_l0 == zero

    # x - x -> 0
    sub_self = SymBinaryOp("-", x, x).simplify()
    assert sub_self == zero

    # (x + y) - y -> x
    add_xy = SymBinaryOp("+", x, y)
    add_sub = SymBinaryOp("-", add_xy, y).simplify()
    assert add_sub == x

    # (y + x) - y -> x
    add_yx = SymBinaryOp("+", y, x)
    add_sub_rev = SymBinaryOp("-", add_yx, y).simplify()
    assert add_sub_rev == x

    # (x * y) // y -> x
    mul_xy = SymBinaryOp("*", x, y)
    mul_div = SymBinaryOp("//", mul_xy, y).simplify()
    assert mul_div == x

    # (y * x) // y -> x
    mul_yx = SymBinaryOp("*", y, x)
    mul_div_rev = SymBinaryOp("//", mul_yx, y).simplify()
    assert mul_div_rev == x


def test_canonical_polynomial_reduction() -> None:
    """Verify canonical polynomial reduction for commutativity, division, and cancellation."""
    b = SymVar("B")
    n = SymVar("N")
    m = SymVar("M")

    # (B * 2) // 2 == B
    b_times_2 = SymBinaryOp("*", b, SymConst(2))
    b_div_2 = SymBinaryOp("//", b_times_2, SymConst(2))
    assert b_div_2.canonical() == b
    assert SymbolicSolver.is_consistent(b_div_2, b)

    # N + M == M + N
    nm = SymBinaryOp("+", n, m)
    mn = SymBinaryOp("+", m, n)
    assert nm.canonical() == mn.canonical()
    assert SymbolicSolver.is_consistent(nm, mn)

    # 2 * B + 3 * B == 5 * B
    term1 = SymBinaryOp("*", SymConst(2), b)
    term2 = SymBinaryOp("*", SymConst(3), b)
    combined = SymBinaryOp("+", term1, term2)
    expected = SymBinaryOp("*", SymConst(5), b)
    assert combined.canonical() == expected.canonical()
    assert SymbolicSolver.is_consistent(combined, expected)


def test_symint_convenience_operations() -> None:
    """Test arithmetic operators on SymInt with automatic canonical comparison."""
    b = SymInt("B")
    assert (b * 2) // 2 == b
    assert b + 0 == b
    assert 0 + b == b
    assert b * 1 == b
    assert 1 * b == b
    assert b * 0 == 0

    n = SymInt("N")
    m = SymInt("M")
    assert n + m == m + n
    assert (n + m) - m == n

    # Division by 1
    assert b // 1 == b

    # Constant folding
    c = SymInt(6) // SymInt(2)
    assert c == 3


def test_symbolic_broadcast_dim_rules() -> None:
    """Test symbolic broadcast rules: _broadcast_dim(SymVar("B"), 1) -> SymVar("B"), etc."""
    b_var = SymVar("B")
    one = SymConst(1)

    # _broadcast_dim(SymVar("B"), 1) -> SymVar("B")
    assert _broadcast_dim(b_var, 1) == b_var
    assert _broadcast_dim(b_var, one) == b_var
    assert _broadcast_dim(1, b_var) == b_var

    # _broadcast_dim(SymVar("B"), SymVar("B")) -> SymVar("B")
    assert _broadcast_dim(b_var, b_var) == b_var

    # Incompatible integer constants raise ShapeMismatchError
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(2, 3)


def test_symbolic_constraint_tracker_and_unification() -> None:
    """Verify SymbolicConstraintTracker records equalities, solves constraints, and unifies dims."""
    tracker = SymbolicConstraintTracker()
    b = SymVar("B")
    t = SymVar("T")

    # Unify B and T
    res = tracker.unify(b, t)
    assert res == b
    assert tracker.is_consistent()

    # Bind B to 16
    tracker.add_constraint(b, 16)
    assert tracker.is_consistent()

    solved = tracker.solve()
    assert solved.get("B") == 16
    assert solved.get("T") == 16

    # Substitute into shape tuple
    shape = (b, t, 128)
    subbed = tracker.substitute(shape)
    assert subbed == (16, 16, 128)

    # Adding a conflicting constraint raises ShapeMismatchError
    with pytest.raises(ShapeMismatchError):
        tracker.add_constraint(t, 32)
    assert not tracker.is_consistent()


def test_symbolic_broadcasting_dynamic_dimensions() -> None:
    """Test broadcasting tensors with dynamic batch sizes and sequence lengths."""
    tracker = SymbolicConstraintTracker()
    b = SymVar("B")
    s = SymVar("S")
    d = 64

    # (B, 1, D) broadcast with (1, S, D) -> (B, S, D)
    shape_a = (b, 1, d)
    shape_b = (1, s, d)
    out_shape = broadcast_shapes(shape_a, shape_b, tracker=tracker)
    assert out_shape == (b, s, d)


def test_attention_head_projection_symbolic() -> None:
    """Test attention head projection shapes: (B, S, H * D) -> (B, H, S, D)."""
    h = 8
    d = 64
    hd = h * d  # 512
    b = SymInt("B")
    s = SymInt("S")

    input_shape = (b, s, hd)
    # Split projection into (B, S, H, D)
    per_head_dim = input_shape[2] // h
    assert per_head_dim == d

    # Transposed to (B, H, S, D)
    attn_heads_shape = (b, h, s, per_head_dim)
    assert attn_heads_shape == (b, h, s, d)


def test_symbolic_constraint_tracker_graph_traversal() -> None:
    """Test SymbolicConstraintTracker traversing a graph to extract constraints."""
    tracker = SymbolicConstraintTracker()
    b = SymVar("B")
    s1 = SymVar("S1")
    s2 = SymVar("S2")

    n1 = MockLogicalNode(op="input", inputs=[], shape=(b, s1, 64))
    n2 = MockLogicalNode(op="input", inputs=[], shape=(b, 64, s2))
    n_matmul = MockLogicalNode(op="matmul", inputs=[n1, n2], shape=(b, s1, s2))
    graph = MockLogicalGraph(nodes=[n1, n2, n_matmul])

    tracker.track_graph(graph)
    assert tracker.is_consistent()

    # Now test add op with broadcasting
    t1 = MockLogicalNode(op="input", inputs=[], shape=(b, 1, 64))
    t2 = MockLogicalNode(op="input", inputs=[], shape=(1, s1, 64))
    t_add = MockLogicalNode(op="add", inputs=[t1, t2], shape=(b, s1, 64))
    graph2 = MockLogicalGraph(nodes=[t1, t2, t_add])
    tracker.track_graph(graph2)
    assert tracker.is_consistent()


def test_symnode_eval_and_free_vars() -> None:
    """Test eval and free_vars across SymConst, SymVar, and SymBinaryOp."""
    x = SymVar("x")
    y = SymVar("y")
    c2 = SymConst(2)

    expr = (x * c2) + y
    assert expr.free_vars() == {"x", "y"}
    val = expr.eval({"x": 10, "y": 5})
    assert val == 25

    # Evaluation with missing variable raises KeyError
    with pytest.raises(KeyError):
        expr.eval({"x": 10})

    # Division by zero in eval raises ZeroDivisionError
    div_zero = SymBinaryOp("//", x, SymConst(0))
    with pytest.raises(ZeroDivisionError):
        div_zero.eval({"x": 10})

    # Unsupported op raises ValueError
    bad_op = SymBinaryOp("%", x, c2)
    with pytest.raises(ValueError):
        bad_op.eval({"x": 10})


def test_symnode_conversion_errors() -> None:
    """Test type error when converting an invalid object to SymNode."""
    with pytest.raises(TypeError):
        SymNode.to_node([1, 2, 3])  # type: ignore[arg-type]


def test_polynomial_edge_cases() -> None:
    """Test Polynomial operations including zero, multiplication, and string representation."""
    p0 = Polynomial.from_const(0)
    assert p0.is_zero()
    assert p0.canonical_str() == "0"
    assert repr(p0) == "Polynomial(0)"

    p3 = Polynomial.from_const(3)
    assert p3.is_const()
    assert p3.get_const() == 3

    px = Polynomial.from_var("x")
    py = Polynomial.from_var("y")

    # Multiplication
    pxy = px.mul(py)
    assert "x" in pxy.canonical_str() and "y" in pxy.canonical_str()

    # Division by zero
    assert px.div_exact(p0) is None

    # Division by constant zero
    assert px.div_exact(Polynomial.from_const(0)) is None

    # Division non-divisible
    p_non = Polynomial.from_const(5)
    p_div = Polynomial.from_const(2)
    assert p_non.div_exact(p_div) is None

    # Inequality comparison
    assert px != "not_a_polynomial"


def test_symnode_direct_operators_and_representations() -> None:
    """Test operators and representations directly on SymNode instances."""
    x = SymVar("x")
    c5 = SymConst(5)

    # SymNode operators
    add_node = x + c5
    assert isinstance(add_node, SymBinaryOp)
    radd_node = 3 + x
    assert isinstance(radd_node, SymBinaryOp)
    sub_node = x - c5
    assert isinstance(sub_node, SymBinaryOp)
    rsub_node = 10 - x
    assert isinstance(rsub_node, SymBinaryOp)
    mul_node = x * c5
    assert isinstance(mul_node, SymBinaryOp)
    rmul_node = 2 * x
    assert isinstance(rmul_node, SymBinaryOp)
    div_node = x // c5
    assert isinstance(div_node, SymBinaryOp)
    rdiv_node = 100 // x
    assert isinstance(rdiv_node, SymBinaryOp)

    # SymConst representations and equality
    assert repr(c5) == "SymConst(5)"
    assert hash(c5) == hash(5)
    assert c5 == 5
    assert c5 == SymConst(5)
    assert c5 != SymConst(6)
    assert c5 != "5"

    # SymVar representations and equality
    assert repr(x) == "SymVar('x')"
    assert hash(x) == hash("x")
    assert x == SymVar("x")
    assert x != SymVar("y")
    assert x != "x"

    # SymBinaryOp representations and equality
    op1 = SymBinaryOp("+", x, c5)
    op2 = SymBinaryOp("+", x, c5)
    op3 = SymBinaryOp("-", x, c5)
    assert repr(op1) == "SymBinaryOp('+', SymVar('x'), SymConst(5))"
    assert hash(op1) == hash(("+", x, c5))
    assert op1 == op2
    assert op1 != op3
    assert op1 != "not_an_op"


def test_polynomial_arithmetic_and_string_formatting() -> None:
    """Test polynomial subtraction, term cancellation, degree formatting, and hash."""
    x = Polynomial.from_var("x")
    y = Polynomial.from_var("y")

    # Term cancellation in sub
    sub_res = x.sub(x)
    assert sub_res.is_zero()

    # Addition cancelling out
    neg_x = Polynomial({((("x", 1),)): -1})
    add_cancel = x.add(neg_x)
    assert add_cancel.is_zero()

    # Higher power formatting: x^2 * y
    p_quad = Polynomial({((("x", 2), ("y", 1))): 3})
    assert "3*x^2*y" in p_quad.canonical_str()

    # Leading negative term and subsequent negative term
    p_neg = Polynomial({((("x", 1),)): -2, (): -5})
    s_neg = p_neg.canonical_str()
    assert s_neg.startswith("-2*x")
    assert " - 5" in s_neg

    # Polynomial division by exact monomial factor with powers
    p_poly = Polynomial({((("x", 2), ("y", 1))): 4})
    p_div = Polynomial({((("x", 1),)): 2})
    p_quot = p_poly.div_exact(p_div)
    assert p_quot is not None
    assert p_quot.terms == {((("x", 1), ("y", 1))): 2}

    # Division where divisor has higher power than dividend
    p_small = Polynomial({((("x", 1),)): 2})
    p_big = Polynomial({((("x", 2),)): 2})
    assert p_small.div_exact(p_big) is None

    # Division where other is general multi-term polynomial
    p_multi = Polynomial({((("x", 1),)): 1, ((("y", 1),)): 1})
    assert p_poly.div_exact(p_multi) is None

    # Polynomial hash
    assert isinstance(hash(p_poly), int)


def test_symbinaryop_simplify_branches() -> None:
    """Test all branches of SymBinaryOp.simplify."""
    c2 = SymConst(2)
    c3 = SymConst(3)
    c0 = SymConst(0)
    c1 = SymConst(1)
    x = SymVar("x")
    y = SymVar("y")

    # Constant folding
    assert SymBinaryOp("+", c2, c3).simplify() == SymConst(5)
    assert SymBinaryOp("-", c3, c2).simplify() == SymConst(1)
    assert SymBinaryOp("*", c2, c3).simplify() == SymConst(6)
    assert SymBinaryOp("//", SymConst(6), c2).simplify() == SymConst(3)

    # Subtraction rules
    assert SymBinaryOp("-", x, c0).simplify() == x
    assert SymBinaryOp("-", SymBinaryOp("+", y, x), x).simplify() == y

    # Floor division rules
    assert SymBinaryOp("//", x, c1).simplify() == x
    assert SymBinaryOp("//", c0, x).simplify() == c0
    assert SymBinaryOp("//", x, x).simplify() == c1
    assert SymBinaryOp("//", SymBinaryOp("*", y, x), x).simplify() == y


def test_poly_to_symnode_conversions() -> None:
    """Test conversion of various polynomial forms back to SymNode."""
    from ml_switcheroo_compiler.ir.shape_system import _poly_to_symnode

    # Zero polynomial
    p0 = Polynomial.from_const(0)
    assert isinstance(_poly_to_symnode(p0), SymConst)
    assert _poly_to_symnode(p0).value == 0

    # Const polynomial
    p7 = Polynomial.from_const(7)
    assert isinstance(_poly_to_symnode(p7), SymConst)
    assert _poly_to_symnode(p7).value == 7

    # Polynomial with negative leading coefficient
    p_neg_lead = Polynomial({((("x", 1),)): -3})
    node_neg = _poly_to_symnode(p_neg_lead)
    assert isinstance(node_neg, SymBinaryOp)
    assert node_neg.op == "-"

    # Polynomial with negative trailing term
    p_sub = Polynomial({((("x", 1),)): 2, (): -4})
    node_sub = _poly_to_symnode(p_sub)
    assert isinstance(node_sub, SymBinaryOp)
    assert node_sub.op == "-"


def test_expression_parser_branches() -> None:
    """Test tokenization and parsing of nested, parenthesized, and unary-minus expressions."""
    from ml_switcheroo_compiler.ir.shape_system import _parse_sym_str, _tokenize_expr

    # Empty string
    assert _tokenize_expr("   ") == []
    node_empty = _parse_sym_str("")
    assert isinstance(node_empty, SymVar)

    # Expression with negative number / unary minus
    node_neg = _parse_sym_str("(-5 + x)")
    assert isinstance(node_neg, SymBinaryOp)

    # Expression with single slash treated as floor div
    tokens = _tokenize_expr("a / b")
    assert "//" in tokens

    # Parenthesized term
    node_paren = _parse_sym_str("(x * 2) // 2")
    assert isinstance(node_paren, SymBinaryOp)

    # Incomplete expression where factor hits end of tokens
    node_incomplete = _parse_sym_str("x + ")
    assert isinstance(node_incomplete, SymBinaryOp)

    # Unparseable fallback
    node_fallback = _parse_sym_str("x y z")
    assert isinstance(node_fallback, SymVar)

    # Identifier starting with underscore
    tokens_under = _tokenize_expr("_batch_size + 1")
    assert "_batch_size" in tokens_under


def test_polynomial_cross_product_cancellation() -> None:
    """Test polynomial multiplication cross-product cancellation (x + 1)(x - 1) -> x^2 - 1."""
    x_plus_1 = Polynomial({((("x", 1),)): 1, (): 1})
    x_minus_1 = Polynomial({((("x", 1),)): 1, (): -1})
    prod = x_plus_1.mul(x_minus_1)
    assert prod.terms == {((("x", 2),)): 1, (): -1}

    # get_const when no constant term exists
    p_no_const = Polynomial.from_var("x")
    assert p_no_const.get_const() == 0

    # Polynomial division by exact 1
    p_const_1 = Polynomial.from_const(1)
    assert _poly_to_symnode(p_const_1) == SymConst(1)


def test_broadcast_dim_symconst_branches() -> None:
    """Test _broadcast_dim with SymConst instances."""
    sc1 = SymConst(1)
    sc4 = SymConst(4)
    sc5 = SymConst(5)

    # SymConst(1) broadcasting
    assert _broadcast_dim(sc1, 4) == 4
    assert _broadcast_dim(4, sc1) == 4

    # SymConst equality
    assert _broadcast_dim(sc4, sc4) == 4

    # Incompatible SymConst
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(sc4, sc5)


def test_symint_canonical_explicit() -> None:
    """Test explicit call to SymInt.canonical()."""
    b = SymInt("B")
    can = b.canonical()
    assert isinstance(can, SymInt)
    assert can == b


def test_track_graph_empty_node_shapes() -> None:
    """Test track_graph with nodes that have None or empty shapes."""
    tracker = SymbolicConstraintTracker()
    n_none = MockLogicalNode(op="custom", inputs=[], shape=())
    graph = MockLogicalGraph(nodes=[n_none])
    tracker.track_graph(graph)
    assert tracker.is_consistent()


def test_polynomial_and_symnode_detailed_branches() -> None:
    """Test detailed branches in Polynomial and SymNode."""
    # Polynomial initialized with 0 coefficient
    p_zero_coeff = Polynomial({((("x", 1),)): 0})
    assert p_zero_coeff.is_zero()

    # Polynomial sub with new terms
    px = Polynomial.from_var("x")
    py = Polynomial.from_var("y")
    pxy = px.sub(py)
    assert pxy.terms == {((("x", 1),)): 1, ((("y", 1),)): -1}

    # Division by constant zero
    p_zero_const = Polynomial.from_const(0)
    assert px.div_exact(p_zero_const) is None

    # Division where coefficient is not divisible
    p5 = Polynomial.from_const(5)
    p2 = Polynomial.from_const(2)
    assert p5.div_exact(p2) is None

    # Division where single monomial coeff not divisible
    p3x = Polynomial({((("x", 1),)): 3})
    p2x = Polynomial({((("x", 1),)): 2})
    assert p3x.div_exact(p2x) is None

    # Division where power in divisor exceeds power in dividend
    px2 = Polynomial({((("x", 2),)): 1})
    assert px.div_exact(px2) is None

    # Polynomial hash
    assert isinstance(hash(px), int)

    # SymBinaryOp simplify division by zero constant
    div_zero_op = SymBinaryOp("//", SymConst(5), SymConst(0))
    simplified_div_zero = div_zero_op.simplify()
    assert isinstance(simplified_div_zero, SymBinaryOp)

    # to_polynomial for binary op operators
    x = SymVar("x")
    y = SymVar("y")
    sub_op = SymBinaryOp("-", x, y)
    assert sub_op.to_polynomial() is not None

    mul_op = SymBinaryOp("*", x, y)
    assert mul_op.to_polynomial() is not None

    # Poly to symnode with constant term 0
    p_empty = Polynomial({})
    assert _poly_to_symnode(p_empty) == SymConst(0)

    # Tokenizer with underscore identifier
    tokens = _tokenize_expr("_dim1 + 1")
    assert "_dim1" in tokens

    # Parentheses unclosed handling in parse_factor
    p_unclosed = _parse_sym_str("(x + 1")
    assert isinstance(p_unclosed, (SymBinaryOp, SymVar))

    # SymInt.canonical
    si = SymInt("x + 0")
    assert isinstance(si.canonical(), SymInt)

    # Constraint tracker root_a / root_b in const_map
    tracker = SymbolicConstraintTracker()
    tracker.record_equality("A", 10)  # root_a has const
    tracker.record_equality("B", 10)  # root_b has const
    tracker.record_equality("A", "B")  # both roots have const, matching
    assert tracker.is_consistent()

    tracker2 = SymbolicConstraintTracker()
    tracker2.record_equality(15, "X")
    tracker2.record_equality(15, "Y")
    tracker2.record_equality("X", "Y")
    assert tracker2.is_consistent()

    # Substitute on registered integer
    tracker2.record_equality(20, 20)
    sub_res = tracker2.substitute(20)
    assert sub_res == 20

    # Track graph with node with input without shape
    n_noshape = MockLogicalNode(op="matmul", inputs=[MockLogicalNode(op="in", inputs=[], shape=())], shape=())
    tracker2.track_graph(MockLogicalGraph(nodes=[n_noshape]))

    # _broadcast_dim where b is SymConst
    sc3 = SymConst(3)
    assert _broadcast_dim(3, sc3) == 3


def test_symnode_base_methods() -> None:
    """Test default implementations on the base SymNode class."""
    base = SymNode()
    assert base.simplify() is base
    assert base.to_polynomial() is None
    assert base.canonical() is base
    with pytest.raises(NotImplementedError):
        base.eval({})
    assert base.free_vars() == set()
    add_res = base + 1
    assert isinstance(add_res, SymBinaryOp)


def test_solver_fallback_non_polynomial() -> None:
    """Test SymbolicSolver fallback when nodes are irreducible to polynomials."""
    x = SymVar("x")
    op1 = SymBinaryOp("%", x, SymConst(2))
    op2 = SymBinaryOp("%", x, SymConst(2))
    assert SymbolicSolver.is_consistent(op1, op2)

    op3 = SymBinaryOp("%", x, SymConst(3))
    assert not SymbolicSolver.is_consistent(op1, op3)


def test_symint_additional_methods_and_comparisons() -> None:
    """Test SymInt right-operators, comparisons with non-SymInt, and hash."""
    x = SymInt("x")
    assert repr(x) == "SymInt(x)"
    assert hash(x) == hash("x")

    # Initializing from existing SymInt
    x_copy = SymInt(x)
    assert x_copy == x
    assert x_copy.canonical() == x

    # Right floor div
    rdiv = 100 // x
    assert isinstance(rdiv, SymInt)

    # Equality against non-symbolic
    assert x != "x"
    assert x != [1, 2]

    # Equality against int
    assert SymInt(42) == 42
    assert SymInt(42) != 43

    # Equality against SymNode
    assert x == SymVar("x")


def test_symbolic_constraint_tracker_exhaustive() -> None:
    """Test all branches in SymbolicConstraintTracker."""
    tracker = SymbolicConstraintTracker()

    # Identical integer equality
    tracker.record_equality(5, 5)
    assert tracker.is_consistent()

    # Contradictory integer equality
    with pytest.raises(ShapeMismatchError):
        tracker.record_equality(5, 10)
    assert not tracker.is_consistent()

    # Fresh tracker for SymConst, digit strings, and unification
    t2 = SymbolicConstraintTracker()
    t2.record_equality(SymConst(4), "4")
    assert t2.is_consistent()

    # Unify with 1, "1", and SymConst(1)
    b = SymVar("B")
    assert t2.unify(b, 1) == b
    assert t2.unify(1, b) == b
    assert t2.unify(b, "1") == b
    assert t2.unify("1", b) == b
    assert t2.unify(b, SymConst(1)) == b

    # Unify resulting in bound constant replacement
    t2.add_constraint(b, 8)
    res = t2.unify(b, SymVar("T"))
    assert res == SymConst(8)

    # Substitute on SymInt, SymNode, int, and string
    si = SymInt("B")
    sub_si = t2.substitute(si)
    assert isinstance(sub_si, SymInt) and sub_si == 8

    sn = SymVar("B")
    sub_sn = t2.substitute(sn)
    assert isinstance(sub_sn, SymConst) and sub_sn.value == 8

    assert t2.substitute(42) == 42
    assert t2.substitute("unbound") == "unbound"

    # Unify where input 'a' is a SymInt
    si_res = t2.unify(si, SymInt("Z"))
    assert isinstance(si_res, SymInt) and si_res == 8

    # Record equality where 'a' is int and 'b' is unbound symbol
    t3 = SymbolicConstraintTracker()
    t3.record_equality(16, "B_dim")
    assert t3.solve().get("B_dim") == 16
    t3.record_equality("C_dim", "B_dim")
    assert t3.solve().get("C_dim") == 16
    t3.record_equality("B_dim", "D_dim")
    assert t3.solve().get("D_dim") == 16

    # Substitute on integer that was registered in tracker
    t3.record_equality(100, 100)
    assert t3.substitute(100) == 100

    # Broadcasting integer dimension with symbol without tracker raises ShapeMismatchError
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(2, "X")
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim("X", 2)

    # But with tracker, it unifies successfully
    t4 = SymbolicConstraintTracker()
    assert _broadcast_dim(2, "X", tracker=t4) == 2
    assert t4.solve().get("X") == 2


def test_shape_and_axis_error_branches() -> None:
    """Test error handling in matmul_shape and normalize_axis."""
    # matmul_shape scalar error
    with pytest.raises(ShapeMismatchError, match="Scalars cannot be matrix multiplied"):
        matmul_shape((), (2, 2))

    with pytest.raises(ShapeMismatchError, match="Scalars cannot be matrix multiplied"):
        matmul_shape((2, 2), ())

    # matmul_shape 1d incompatible
    with pytest.raises(ValueError, match="Incompatible 1D dot product shapes"):
        matmul_shape((3,), (4,))

    # matmul_shape 2d incompatible
    with pytest.raises(ShapeMismatchError, match="Incompatible 2D matmul shapes"):
        matmul_shape((2, 3), (4, 5))

    # matmul_shape batched incompatible
    with pytest.raises(ShapeMismatchError, match="Incompatible inner dimensions"):
        matmul_shape((2, 3, 4), (2, 5, 6))

    # normalize_axis out of bounds
    with pytest.raises(ShapeMismatchError, match="out of bounds"):
        normalize_axis(5, 3)

    with pytest.raises(ShapeMismatchError, match="out of bounds"):
        normalize_axis(-5, 3)

    # normalize_axis invalid type
    with pytest.raises(TypeError, match="Invalid type for axis"):
        normalize_axis("0", 3)  # type: ignore[arg-type]

    # normalize_axis list input
    assert normalize_axis([0, -1], 3) == (0, 2)


def test_load_algebraic_rules_invalid_yaml(tmp_path) -> None:
    """Test load_algebraic_rules when YAML does not contain a dictionary."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not\n- a\n- dict\n", encoding="utf-8")
    rules = load_algebraic_rules(str(bad_yaml))
    assert isinstance(rules, AlgebraicRuleSetModel)
    assert len(rules.rules) == 0


def test_shape_system_all_remaining_coverage_branches() -> None:
    """Test all remaining lines and branches in shape_system.py."""
    from unittest.mock import patch

    # 1. Line 142: is_const on zero polynomial
    poly_zero = Polynomial({})
    assert poly_zero.is_const() is True

    # 2. Line 227 & 231 & 252 & 288 & 341: Polynomial div_exact and canonical_str and __str__
    px = Polynomial.from_var("x")
    # 227: self == other
    assert px.div_exact(px) == Polynomial.from_const(1)

    # 231: other is const 0
    poly_const_zero = Polynomial({})
    poly_const_zero.terms[()] = 0
    assert px.div_exact(poly_const_zero) is None

    # 252: mono_dict[var] == 0 in division
    p_xy = px.mul(Polynomial.from_var("y"))
    assert p_xy.div_exact(px) == Polynomial.from_var("y")

    # 288: canonical_str sign = " + "
    p_add = px.add(Polynomial.from_const(2))
    assert " + " in p_add.canonical_str()

    # 341: __str__
    assert str(px) == "x"

    # 3. SymBinaryOp branches: 747->774, 750->774, 771->774, 789, 819, 826
    # 747->774: op == "-" where left is not SymBinaryOp("+")
    s_sub = SymBinaryOp("-", SymVar("a"), SymVar("b"))
    assert s_sub.simplify() is not None

    # 750->774: op == "-" where left is (a + b) and right is c
    s_sub_nomatch = SymBinaryOp("-", SymBinaryOp("+", SymVar("a"), SymVar("b")), SymVar("c"))
    assert s_sub_nomatch.simplify() is not None

    # 771->774: op == "//" where left is (a * b) and right is c
    s_div_nomatch = SymBinaryOp("//", SymBinaryOp("*", SymVar("a"), SymVar("b")), SymVar("c"))
    assert s_div_nomatch.simplify() is not None

    # 789: to_polynomial returns None if left or right returns None
    class IrreducibleNode(SymNode):
        def simplify(self):
            return self

        def canonical(self):
            return self

        def to_polynomial(self):
            return None

        def eval(self, env):
            return 0

        def free_vars(self):
            return set()

    s_irred = SymBinaryOp("+", IrreducibleNode(), SymConst(1))
    assert s_irred.to_polynomial() is None

    # 819 & 826: eval for "-" and "//"
    assert SymBinaryOp("-", SymConst(5), SymConst(2)).eval({}) == 3
    assert SymBinaryOp("//", SymConst(6), SymConst(2)).eval({}) == 3

    # 4. _poly_to_symnode line 925:
    assert _poly_to_symnode(Polynomial({})) == SymConst(0)
    p_power0 = Polynomial({})
    p_power0.terms[(("x", 0),)] = 1
    assert _poly_to_symnode(p_power0) is not None

    # 5. _tokenize_sym_str line 982: unexpected characters
    tokens = _tokenize_expr("a @ 2 $ 3")
    assert "a" in tokens

    # 6. _parse_sym_str lines 1060-1061: syntax error fallback
    with patch("ml_switcheroo_compiler.ir.shape_system.SymBinaryOp.__init__", side_effect=ValueError("forced error")):
        assert _parse_sym_str("1 + 2") == SymVar("1 + 2")

    # 7. SymInt.simplify line 1096:
    s_int = SymInt("2 + 3")
    assert s_int.simplify().node == SymConst(5)

    # 8. Tracker unify lines 1352, 1359, 1375:
    tracker = SymbolicConstraintTracker()
    tracker.record_equality("10", "10")  # lines 1352, 1359 (isdigit keys)
    tracker.record_equality("S", SymConst(7))  # line 1359: isinstance(b, SymConst)
    tracker.record_equality("T", "T")  # line 1375->exit: root_a == root_b and const_a is None
    tracker.record_equality("A", "B")  # line 1375->exit (chosen_const is None)
    # Line 1439->1437: solve() where symbol has no constant mapping
    sol = tracker.solve()
    assert "A" not in sol

    # Lines 1466-1468: substitute with int
    tracker_const = SymbolicConstraintTracker()
    tracker_const.record_equality("K", 42)
    assert tracker_const.substitute(42) == 42
    assert tracker_const.substitute("K") == 42

    # Lines 1486->1493 and 1491->1493: track_graph with empty shapes
    class MockGraph:
        def __init__(self, nodes):
            self.nodes = nodes

    n_mat = MockLogicalNode("matmul", [MockLogicalNode("in1", [], ()), MockLogicalNode("in2", [], ())], (1,))
    n_add = MockLogicalNode("add", [MockLogicalNode("in1", [], None), MockLogicalNode("in2", [], None)], (1,))
    tracker_graph = SymbolicConstraintTracker()
    tracker_graph.track_graph(MockGraph([n_mat, n_add]))

    # Line 1606: broadcast_dimension where SymbolicSolver.is_consistent is True
    from ml_switcheroo_compiler.ir.shape_system import _broadcast_dim

    assert _broadcast_dim("N + 1", "1 + N") == "N + 1"
