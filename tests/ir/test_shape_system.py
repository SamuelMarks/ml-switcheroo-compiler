"""Exhaustive tests for symbolic and dynamic shape inference system."""

import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, TensorSpec
from ml_switcheroo_compiler.ir.shape_system import (
    ShapeMismatchError,
    ShapeTracker,
    SymbolicConstraintTracker,
    SymInt,
    SymVar,
    _broadcast_dim,
    _from_str_shape,
    _matmul_shape_1d,
    _matmul_shape_2d,
    _to_str_shape,
    matmul_shape,
)


def test_symint_eq() -> None:
    """Test the symint eq behavior."""
    s1 = SymInt("A")
    assert s1 != "A"
    assert s1 == SymInt("A")


def test_symint_canonical_and_rsub() -> None:
    """Test canonical and __rsub__ on SymInt."""
    s = SymInt("x + 1")
    canon = s.canonical()
    assert canon is not None

    rsub_res = 10 - SymInt("x")
    assert isinstance(rsub_res, SymInt)
    assert rsub_res.expr == "(10 - x)"


def test_to_str_shape_and_from_str_shape() -> None:
    """Test string conversions for shapes."""
    sym_var = SymVar("batch")
    sym_int = SymInt(sym_var)
    shape_mixed = (sym_int, sym_var, 16, "dim")

    str_shape = _to_str_shape(shape_mixed)
    assert str_shape == ("batch", "batch", 16, "dim")

    roundtrip = _from_str_shape(("128", "channel", 3))
    assert roundtrip[0] == 128
    assert isinstance(roundtrip[1], SymInt)
    assert roundtrip[1].name == "channel"
    assert roundtrip[2] == 3


def test_shape_tracker_infer_elementwise() -> None:
    """Test ShapeTracker elementwise inference."""
    assert ShapeTracker.infer_elementwise([]) == ()

    t1 = TensorSpec(shape=(2, "N", 1), dtype=DType.Float32)
    t2 = TensorSpec(shape=(1, "N", 4), dtype=DType.Float32)
    out = ShapeTracker.infer_elementwise([t1, t2])
    assert out[0] == 2
    assert isinstance(out[1], SymInt)
    assert out[2] == 4


def test_shape_tracker_infer_matmul() -> None:
    """Test ShapeTracker matmul inference."""
    t1 = TensorSpec(shape=(2, 3), dtype=DType.Float32)
    t2 = TensorSpec(shape=(3, 4), dtype=DType.Float32)
    out = ShapeTracker.infer_matmul(t1, t2)
    assert out == (2, 4)


def test_shape_tracker_update_from_feedback() -> None:
    """Test feedback updating with and without tracker."""
    tracker = SymbolicConstraintTracker()

    telemetry_dict = {"shapes": {"node_a": [2, 4], "node_b": (8,)}}
    res = ShapeTracker.update_from_feedback(telemetry_dict, tracker=tracker)
    assert res == {"node_a": (2, 4), "node_b": (8,)}

    flat_telemetry = {"node_c": [1, 3], "ignored_scalar": 42}
    res_flat = ShapeTracker.update_from_feedback(flat_telemetry, tracker=None)
    assert res_flat == {"node_c": (1, 3)}


def test_shape_tracker_resolve_dynamic_bounds() -> None:
    """Test resolve_dynamic_bounds across graph nodes."""
    tracker = SymbolicConstraintTracker()
    graph = IRGraph(name="test_graph")

    n1 = IRNode(
        id="n1",
        op_type="Input",
        shape_metadata=(SymInt("B"), SymVar("H"), "W", 16, "32"),
    )
    n2 = IRNode(
        id="n2",
        op_type="Input",
        shape_metadata=None,
    )
    n2.shape = (SymInt("C"),)

    n3 = IRNode(
        id="n3",
        op_type="Input",
        shape_metadata=(10,),
    )

    n_invalid_meta = IRNode(
        id="n_invalid",
        op_type="Input",
        shape_metadata=123,
    )

    graph.nodes = {"n1": n1, "n2": n2, "n3": n3, "n_invalid": n_invalid_meta}

    telemetry = {
        "n1": [4, 8, 16, 16, 32],
        "n2": [32],
        "n_invalid": [123],
    }

    bounds = ShapeTracker.resolve_dynamic_bounds(graph, telemetry, tracker=tracker)
    assert bounds["B"] == 4
    assert bounds["H"] == 8
    assert bounds["W"] == 16
    assert bounds["C"] == 32
    assert n1.shape_metadata == (4, 8, 16, 16, 32)
    assert n2.shape == (32,)

    # Test when tracker is None with fresh graph
    graph2 = IRGraph(name="test_graph2")
    n1_fresh = IRNode(
        id="n1",
        op_type="Input",
        shape_metadata=(SymInt("B"), SymVar("H"), "W", 16, "32"),
    )
    graph2.nodes = {"n1": n1_fresh}
    bounds_no_tracker = ShapeTracker.resolve_dynamic_bounds(
        graph2,
        {"n1": [5, 9, 17, 16, 32]},
        tracker=None,
    )
    assert bounds_no_tracker["B"] == 5
    assert bounds_no_tracker["H"] == 9
    assert bounds_no_tracker["W"] == 17

    # Test when node has only shape (no shape_metadata attribute)
    class NodeOnlyShape:
        """Mock node having only shape."""

        def __init__(self, node_id: str, shape: tuple[object, ...]) -> None:
            """Initialize mock node.

            Args:
                node_id (str): Node identifier.
                shape (tuple[object, ...]): Node shape.
            """
            self.id = node_id
            self.shape = shape

    class NodeOnlyShapeMetadata:
        """Mock node having only shape_metadata."""

        def __init__(self, node_id: str, shape_metadata: tuple[object, ...]) -> None:
            """Initialize mock node.

            Args:
                node_id (str): Node identifier.
                shape_metadata (tuple[object, ...]): Node shape metadata.
            """
            self.id = node_id
            self.shape_metadata = shape_metadata

    class GraphWithListNodes:
        """Mock graph containing a list of nodes."""

        def __init__(self, nodes: list[object]) -> None:
            """Initialize mock graph.

            Args:
                nodes (list[object]): List of nodes.
            """
            self.nodes = nodes

    n_only_shape = NodeOnlyShape("n_s", (SymInt("S"),))
    n_only_meta = NodeOnlyShapeMetadata("n_m", (SymInt("M"), SymInt("K")))
    list_graph = GraphWithListNodes([n_only_shape, n_only_meta])

    bounds_list = ShapeTracker.resolve_dynamic_bounds(list_graph, {"n_s": [12], "n_m": [7]}, tracker=tracker)
    assert bounds_list["S"] == 12
    assert bounds_list["M"] == 7
    assert bounds_list["K"] == 1
    assert n_only_shape.shape == (12,)
    assert n_only_meta.shape_metadata == (7, 1)


def test_broadcast_dim_unify_tracker_and_fallbacks() -> None:
    """Test _broadcast_dim fallback branches."""
    tracker = SymbolicConstraintTracker()
    dim_a = SymVar("dim_a")
    dim_b = SymVar("dim_b")

    # With tracker
    unified = _broadcast_dim(dim_a, dim_b, tracker=tracker)
    assert unified is not None

    # Without tracker and both are symbolic/str -> returns a
    no_tracker = _broadcast_dim(dim_a, dim_b, tracker=None)
    assert no_tracker == dim_a

    # Without tracker and one is incompatible int and other is str -> raises ShapeMismatchError
    with pytest.raises(ShapeMismatchError, match="Incompatible dimensions for broadcasting"):
        _broadcast_dim(2, "X", tracker=None)


def test_matmul_shapes_errors_and_branches() -> None:
    """Test error handling and edge branches in matmul_shape."""
    with pytest.raises(ShapeMismatchError, match="Scalars cannot be matrix multiplied"):
        matmul_shape((), (2, 3))

    with pytest.raises(ValueError, match="Incompatible 1D dot product shapes"):
        _matmul_shape_1d((3,), (4,))

    assert _matmul_shape_1d((3,), (3,)) == ()

    with pytest.raises(ShapeMismatchError, match="Incompatible 2D matmul shapes"):
        _matmul_shape_2d((2, 3), (4, 5))

    assert _matmul_shape_2d((2, 3), (3, 4)) == (2, 4)

    # 3D @ 3D
    assert matmul_shape((2, 2, 3), (2, 3, 4)) == (2, 2, 4)

    # 1D @ 3D (len(shape_a) == 1, len(shape_b) > 2) -> branch 1843->1845
    assert matmul_shape((3,), (2, 3, 4)) == (2, 4)

    # 3D @ 1D (len(shape_a) > 2, len(shape_b) == 1) -> branch 1845->1848
    assert matmul_shape((2, 4, 3), (3,)) == (2, 4)


def test_symnode_eval_fallback_and_polynomial() -> None:
    """Test Polynomial.eval and SymNode.eval fallback evaluation."""
    from ml_switcheroo_compiler.ir.shape_system import Polynomial, SymNode

    # Test Polynomial eval directly
    p = Polynomial()
    p.terms[(("x", 2),)] = 3
    p.terms[(("y", 1),)] = 5
    p.terms[()] = 7
    # 3 * x^2 + 5 * y + 7 with x=2, y=3: 3*4 + 5*3 + 7 = 12 + 15 + 7 = 34
    res = p.eval({"x": 2, "y": 3})
    assert res == 34

    with pytest.raises(KeyError, match="Variable 'y' not bound"):
        p.eval({"x": 2})

    # Test SymNode fallback evaluation via to_polynomial
    class PolySymNode(SymNode):
        def to_polynomial(self) -> Polynomial:
            return p

    node = PolySymNode()
    assert node.eval({"x": 2, "y": 3}) == 34

    # Test SymNode without polynomial raising NotImplementedError
    class BareSymNode(SymNode):
        pass

    bare = BareSymNode()
    with pytest.raises(NotImplementedError, match="Evaluation not implemented"):
        bare.eval({"x": 1})


def test_symnode_fallback_eval_branches() -> None:
    """Test all operator branches in SymNode.eval fallback."""
    from ml_switcheroo_compiler.ir.shape_system import SymConst, SymNode

    class SimplifyDummy(SymNode):
        def simplify(self) -> SymNode:
            return SymConst(100)

    assert SimplifyDummy().eval({}) == 100

    class BinaryDummy(SymNode):
        def __init__(self, op: str, left: SymNode, right: SymNode) -> None:
            self.op = op
            self.left = left
            self.right = right

    ops_expected = [
        ("+", 6),
        ("-", 2),
        ("*", 8),
        ("//", 2),
        ("/", 2),
        ("%", 0),
        ("**", 16),
        ("min", 2),
        ("max", 4),
        ("==", 0),
        ("!=", 1),
        ("<", 0),
        ("<=", 0),
        (">", 1),
        (">=", 1),
    ]
    for op, expected in ops_expected:
        node = BinaryDummy(op, SymConst(4), SymConst(2))
        assert node.eval({}) == expected

    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        BinaryDummy("//", SymConst(4), SymConst(0)).eval({})

    with pytest.raises(ZeroDivisionError, match="Modulo by zero"):
        BinaryDummy("%", SymConst(4), SymConst(0)).eval({})

    class UnaryDummy(SymNode):
        def __init__(self, op: str, operand: SymNode) -> None:
            self.op = op
            self.operand = operand

    uops_expected = [
        ("abs", 5),
        ("-", 5),
        ("floor", -5),
        ("ceil", -5),
    ]
    for uop, expected in uops_expected:
        unode = UnaryDummy(uop, SymConst(-5))
        assert unode.eval({}) == expected

    # Unknown binary op falling through to unary / not implemented
    class UnknownBin(SymNode):
        def __init__(self) -> None:
            self.op = "unknown_op"
            self.left = SymConst(1)
            self.right = SymConst(2)

    with pytest.raises(NotImplementedError, match="Evaluation not implemented"):
        UnknownBin().eval({})

    # Unknown unary op falling through to not implemented
    class UnknownUnary(SymNode):
        def __init__(self) -> None:
            self.op = "unknown_unary"
            self.operand = SymConst(1)

    with pytest.raises(NotImplementedError, match="Evaluation not implemented"):
        UnknownUnary().eval({})


def test_symbolic_simplification_and_dunder_branches() -> None:
    """Test missing branch simplifications for min, max, div, pow, mod, piecewise."""
    from ml_switcheroo_compiler.ir.shape_system import (
        SymBinaryOp,
        SymConst,
        SymNode,
        SymPiecewise,
        SymUnaryOp,
        SymVar,
    )

    x = SymVar("x")
    y = SymVar("y")

    # 1. __rmod__ and __rpow__
    rmod_node = 10 % x
    assert isinstance(rmod_node, SymBinaryOp)
    assert rmod_node.op == "%"

    rpow_node = 2**x
    assert isinstance(rpow_node, SymBinaryOp)
    assert rpow_node.op == "**"

    # 2. Division simplification with constant multiple (lines 913-915)
    # s_left.right is const: (x * 6) // 3
    div_node = SymBinaryOp("//", SymBinaryOp("*", x, SymConst(6)), SymConst(3)).simplify()
    assert div_node == SymBinaryOp("*", SymConst(2), x)

    # s_left.left is const: (6 * x) // 3
    div_node_left = SymBinaryOp("//", SymBinaryOp("*", SymConst(6), x), SymConst(3)).simplify()
    assert div_node_left == SymBinaryOp("*", SymConst(2), x)

    # 3. min simplifications (lines 930-943)
    # nested min
    assert (SymBinaryOp("min", SymBinaryOp("min", x, y), x)).simplify() == SymBinaryOp("min", x, y)
    assert (SymBinaryOp("min", SymBinaryOp("min", y, x), x)).simplify() == SymBinaryOp("min", y, x)
    assert (SymBinaryOp("min", x, SymBinaryOp("min", x, y))).simplify() == SymBinaryOp("min", x, y)
    assert (SymBinaryOp("min", x, SymBinaryOp("min", y, x))).simplify() == SymBinaryOp("min", y, x)

    # min with +
    assert (SymBinaryOp("min", SymBinaryOp("+", x, SymConst(5)), x)).simplify() == x
    assert (SymBinaryOp("min", SymBinaryOp("+", SymConst(5), x), x)).simplify() == x
    assert (SymBinaryOp("min", x, SymBinaryOp("+", x, SymConst(5)))).simplify() == x
    assert (SymBinaryOp("min", x, SymBinaryOp("+", SymConst(5), x))).simplify() == x

    # min with -
    assert (SymBinaryOp("min", SymBinaryOp("-", x, SymConst(5)), x)).simplify() == SymBinaryOp("-", x, SymConst(5))
    assert (SymBinaryOp("min", x, SymBinaryOp("-", x, SymConst(5)))).simplify() == SymBinaryOp("-", x, SymConst(5))

    # 4. max simplifications (lines 950-963)
    # nested max
    assert (SymBinaryOp("max", SymBinaryOp("max", x, y), x)).simplify() == SymBinaryOp("max", x, y)
    assert (SymBinaryOp("max", SymBinaryOp("max", y, x), x)).simplify() == SymBinaryOp("max", y, x)
    assert (SymBinaryOp("max", x, SymBinaryOp("max", x, y))).simplify() == SymBinaryOp("max", x, y)
    assert (SymBinaryOp("max", x, SymBinaryOp("max", y, x))).simplify() == SymBinaryOp("max", y, x)

    # max with +
    assert (SymBinaryOp("max", SymBinaryOp("+", x, SymConst(5)), x)).simplify() == SymBinaryOp("+", x, SymConst(5))
    assert (SymBinaryOp("max", SymBinaryOp("+", SymConst(5), x), x)).simplify() == SymBinaryOp("+", x, SymConst(5))
    assert (SymBinaryOp("max", x, SymBinaryOp("+", x, SymConst(5)))).simplify() == SymBinaryOp("+", x, SymConst(5))
    assert (SymBinaryOp("max", x, SymBinaryOp("+", SymConst(5), x))).simplify() == SymBinaryOp("+", x, SymConst(5))

    # max with -
    assert (SymBinaryOp("max", SymBinaryOp("-", x, SymConst(5)), x)).simplify() == x
    assert (SymBinaryOp("max", x, SymBinaryOp("-", x, SymConst(5)))).simplify() == x

    # 5. SymUnaryOp.simplify returning SymUnaryOp (line 1120)
    u_simp = SymUnaryOp("abs", x).simplify()
    assert isinstance(u_simp, SymUnaryOp)

    # 6. SymPiecewise bool conditions, __eq__, __hash__ (lines 1214, 1242, 1252)
    pw_true = SymPiecewise([(True, SymConst(42))], default=SymConst(0))
    assert pw_true.eval({}) == 42
    pw_false = SymPiecewise([(False, SymConst(42))], default=SymConst(0))
    assert pw_false.eval({}) == 0

    pw1 = SymPiecewise([(True, SymConst(42))], default=SymConst(0))
    pw2 = SymPiecewise([(True, SymConst(42))], default=SymConst(0))
    assert pw1 == pw2
    assert hash(pw1) == hash(pw2)
    assert pw1 != 123

    # Line 913: UnsimplifyingMul to keep right operand as const
    class UnsimplifyingMul(SymBinaryOp):
        def simplify(self) -> SymNode:
            return self

    raw_mul = UnsimplifyingMul("*", x, SymConst(6))
    assert SymBinaryOp("//", raw_mul, SymConst(3)).simplify() == SymBinaryOp("*", SymConst(2), x)

    # Division remainder != 0 (line 914->965 false branch)
    assert SymBinaryOp("//", SymBinaryOp("*", x, SymConst(5)), SymConst(3)).simplify() == SymBinaryOp("//", SymBinaryOp("*", SymConst(5), x), SymConst(3))

    # Lines 935, 938, 955, 958 false branches: + with negative const
    assert SymBinaryOp("min", SymBinaryOp("+", x, SymConst(-5)), x).simplify() == SymBinaryOp("-", x, SymConst(5))
    assert SymBinaryOp("min", x, SymBinaryOp("+", x, SymConst(-5))).simplify() == SymBinaryOp("-", x, SymConst(5))
    assert SymBinaryOp("max", SymBinaryOp("+", x, SymConst(-5)), x).simplify() == x
    assert SymBinaryOp("max", x, SymBinaryOp("+", x, SymConst(-5))).simplify() == x

    # Line 1118: SymUnaryOp '-' with SymConst
    assert SymUnaryOp("-", SymConst(7)).simplify() == SymConst(-7)

    # 7. Remaining partial branches (935->937, 938->940, 945->965, 955->957, 958->960, 1118->1120, 1214->1216)
    z = SymVar("z")
    assert SymBinaryOp("min", SymBinaryOp("+", x, y), z).simplify() == SymBinaryOp("min", SymBinaryOp("+", x, y), z)
    assert SymBinaryOp("min", z, SymBinaryOp("+", x, y)).simplify() == SymBinaryOp("min", z, SymBinaryOp("+", x, y))
    assert SymBinaryOp("max", SymBinaryOp("+", x, y), z).simplify() == SymBinaryOp("max", SymBinaryOp("+", x, y), z)
    assert SymBinaryOp("max", z, SymBinaryOp("+", x, y)).simplify() == SymBinaryOp("max", z, SymBinaryOp("+", x, y))

    # Binary op not in (+, -, *, //, %, min, max), e.g. '=='
    assert SymBinaryOp("==", x, y).simplify() == SymBinaryOp("==", x, y)

    # SymUnaryOp unknown op with SymConst operand
    u_cust = SymUnaryOp("custom_unary", SymConst(5)).simplify()
    assert u_cust.op == "custom_unary" and u_cust.operand == SymConst(5)

    # SymPiecewise cond not callable, SymNode, or bool
    # Boolean conditions (True and False) hitting both 1215->1216 (True) and 1215->1216 (False)
    pw_bool_true = SymPiecewise([(True, SymConst(42))], default=SymConst(0))
    assert pw_bool_true.eval({}) == 42
    pw_bool_false = SymPiecewise([(False, SymConst(42))], default=SymConst(0))
    assert pw_bool_false.eval({}) == 0
    # Fall-through when cond is neither callable, SymNode, nor bool
    pw_int = SymPiecewise([(1, SymConst(42))], default=SymConst(0))
    assert pw_int.eval({}) == 0
