"""Exhaustive tests for operator shape inference grounding across linear algebra, searchsorted, sorting, and foreign calls."""

from __future__ import annotations

from ml_switcheroo_compiler.foreign import ForeignCall
from ml_switcheroo_compiler.ops.linalg import CustomLinearSolve, CustomRoot, Vecdot
from ml_switcheroo_compiler.ops.random_ops.core import Rademacher
from ml_switcheroo_compiler.ops.shape.pad_and_tile import Searchsorted, SortComplex
from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Isin, Issubdtype


class MockTensor:
    """Mock tensor object holding shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock tensor with given shape.

        Args:
            shape (tuple[int, ...]): Tensor dimensions.
        """
        self.shape = shape


def test_foreign_call_shape_introspection() -> None:
    """Test ForeignCall.infer_shape across output_shape, callables, subgraphs, and fallbacks."""
    op = ForeignCall()

    # 1. Explicit output_shape kwarg
    assert op.infer_shape(output_shape=(2, 3, 4)) == (2, 3, 4)
    assert op.infer_shape(output_shape=5) == (5,)

    # 2. Object with .shape
    assert op.infer_shape(MockTensor((8, 16))) == (8, 16)

    # 3. Object with .output_shape / .output_shapes
    class MockExternalModule:
        output_shape = (4, 8)

    class MockMultiOutputModule:
        output_shapes = [(3, 5), (2, 2)]

    assert op.infer_shape(MockExternalModule()) == (4, 8)
    assert op.infer_shape(MockMultiOutputModule()) == (3, 5)

    # 4. Bound subgraph introspection
    class MockNode:
        shape_metadata = (1, 64)

    class MockSubgraph:
        outputs = ["out_0"]
        nodes = {"out_0": MockNode()}

    class MockGraphContainer:
        subgraph = MockSubgraph()

    assert op.infer_shape(MockGraphContainer()) == (1, 64)

    # 5. Callable with return annotation
    def annotated_fn(x: int) -> tuple[int, int]:
        return (x, x)

    assert op.infer_shape(annotated_fn) == (int, int)

    # 6. Fallback empty
    assert op.infer_shape() == ()
    assert op.infer_shape(lambda x: x) == ()


def test_searchsorted_and_sort_complex_shape() -> None:
    """Test Searchsorted and SortComplex shape inference with 1D, 2D, and batched broadcasting."""
    ss = Searchsorted()
    sc = SortComplex()

    # Empty inputs
    assert ss.infer_shape() == ()
    assert sc.infer_shape() == ()

    # Searchsorted with 1D sequence and 1D / 2D values
    seq_1d = MockTensor((10,))
    val_1d = MockTensor((5,))
    assert ss.infer_shape(seq_1d, val_1d) == (5,)

    val_2d = MockTensor((3, 7))
    assert ss.infer_shape(seq_1d, val_2d) == (3, 7)

    # Searchsorted with multidimensional sequence and broadcasted values
    seq_batched = MockTensor((2, 1, 10))
    val_batched = MockTensor((1, 4, 8))
    # Leading batch dims (2, 1) and (1, 4) broadcast to (2, 4) + (8,)
    assert ss.infer_shape(seq_batched, val_batched) == (2, 4, 8)

    # SortComplex preserves input shape
    inp_2d = MockTensor((4, 5))
    assert sc.infer_shape(inp_2d) == (4, 5)

    inp_4d = MockTensor((2, 3, 8, 8))
    assert sc.infer_shape(inp_4d) == (2, 3, 8, 8)


def test_linalg_and_random_shape_grounding() -> None:
    """Test Vecdot, CustomLinearSolve, CustomRoot, and Rademacher shape inference."""
    # 1. Vecdot contracts specified axis (-1 by default)
    vecdot = Vecdot()
    assert vecdot.infer_shape() == ()

    # 1D @ 1D -> scalar ()
    x1 = MockTensor((4,))
    y1 = MockTensor((4,))
    assert vecdot.infer_shape(x1, y1) == ()

    # Batched vecdot (2, 3, 5) and (2, 1, 5) along axis -1 -> (2, 3)
    x_batch = MockTensor((2, 3, 5))
    y_batch = MockTensor((2, 1, 5))
    assert vecdot.infer_shape(x_batch, y_batch, axis=-1) == (2, 3)

    # 2. CustomLinearSolve: solves AX = B
    solve = CustomLinearSolve()
    assert solve.infer_shape() == ()

    # A: (2, 4, 4), B: (2, 4) -> X: (2, 4)
    mat_a = MockTensor((2, 4, 4))
    rhs_b1 = MockTensor((2, 4))
    assert solve.infer_shape(mat_a, rhs_b1) == (2, 4)

    # A: (2, 4, 4), B: (2, 4, 3) -> X: (2, 4, 3)
    rhs_b2 = MockTensor((2, 4, 3))
    assert solve.infer_shape(mat_a, rhs_b2) == (2, 4, 3)

    # 3. CustomRoot: matches initial guess x0
    croot = CustomRoot()
    assert croot.infer_shape() == ()
    x0 = MockTensor((10, 2))
    assert croot.infer_shape(None, x0) == (10, 2)
    assert croot.infer_shape(x0) == (10, 2)

    # 4. Rademacher: output sample shape
    rad = Rademacher()
    assert rad.infer_shape() == ()
    assert rad.infer_shape((4, 8)) == (4, 8)
    assert rad.infer_shape(shape=[2, 3]) == (2, 3)
    assert rad.infer_shape(5) == (5,)


def test_logical_reduction_ops_shape() -> None:
    """Test Issubdtype, Isin, and Ediff1d shape inference."""
    issub = Issubdtype()
    isin_op = Isin()
    ediff = Ediff1d()

    # Issubdtype always returns scalar ()
    assert issub.infer_shape() == ()
    assert issub.infer_shape("float32", "floating") == ()

    # Isin preserves element shape
    el = MockTensor((3, 4, 5))
    test_el = MockTensor((10,))
    assert isin_op.infer_shape(el, test_el) == (3, 4, 5)

    # Ediff1d produces 1D differences
    assert ediff.infer_shape() == (0,)

    ary = MockTensor((5,))
    # Without to_end/to_begin: len(5) - 1 == 4
    assert ediff.infer_shape(ary) == (4,)

    # With to_end and to_begin additions
    assert ediff.infer_shape(ary, to_end=[99], to_begin=[11, 22]) == (7,)
