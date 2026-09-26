"""Unit tests for miscellaneous, shape, creation, activation, distributed, and logical ops shape inference."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.foreign import ForeignCall
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace
from ml_switcheroo_compiler.ops.distributed_ops import Outfeed, Pmax, Pmin, Ppermute, Pshuffle, PsumScatter, Pswapaxes
from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, Squareplus
from ml_switcheroo_compiler.ops.shape.pad_and_tile import (
    Flatnonzero,
    Lexsort,
    Nonzero,
    Percentile,
    Quantile,
    RavelMultiIndex,
    Repeat,
    Searchsorted,
    SortComplex,
    Tile,
    Unique,
)
from ml_switcheroo_compiler.ops.shape.slicing import IndexInDim, UpdateSlice
from ml_switcheroo_compiler.ops.unary.logical import (
    Ediff1d,
    Iscomplex,
    Iscomplexobj,
    Isin,
    Isreal,
    Isrealobj,
    Issubdtype,
)


def _make_tensor(shape: tuple[int, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[int, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


def _sample_ast_reflected_func(x: object) -> object:
    """Sample function for AST shape reflection.

    Args:
        x (object): Input tensor.

    Returns:
        object: Result tuple.
    """
    return (10, 20)


def test_pad_and_tile_shapes() -> None:
    """Verify shape inference for pad_and_tile miscellaneous operations."""
    op_fnz = Flatnonzero()
    op_lex = Lexsort()
    op_nz = Nonzero()
    op_perc = Percentile()
    op_quant = Quantile()
    op_ravel = RavelMultiIndex()
    op_rep = Repeat()
    op_ss = Searchsorted()
    op_sc = SortComplex()
    op_tile = Tile()
    op_unique = Unique()

    t2 = _make_tensor((4, 8))

    # Flatnonzero and Nonzero
    assert isinstance(op_fnz.infer_shape(t2)[0], SymVar)
    nz_shapes = op_nz.infer_shape(t2)
    assert len(nz_shapes) == 2
    assert isinstance(nz_shapes[0][0], SymVar)

    # Lexsort
    assert op_lex.infer_shape([t2, t2]) == (4, 8)

    # Percentile and Quantile
    q = _make_tensor((3,))
    assert op_perc.infer_shape(t2, q, axis=1) == (3, 4)
    assert op_quant.infer_shape(t2, q, axis=0) == (3, 8)

    # RavelMultiIndex
    assert op_ravel.infer_shape(([t2, t2], (10, 20))) == (4, 8)

    # Repeat
    assert op_rep.infer_shape(t2, 2, axis=0) == (8, 8)
    assert op_rep.infer_shape(t2, 3, axis=1) == (4, 24)
    assert op_rep.infer_shape(t2, 2, axis=None) == (64,)

    # Searchsorted
    v = _make_tensor((2, 5))
    seq = _make_tensor((2, 10))
    assert op_ss.infer_shape(seq, v) == (2, 5)

    # SortComplex
    assert op_sc.infer_shape(t2) == (4, 8)

    # Tile
    assert op_tile.infer_shape(t2, (2, 3)) == (8, 24)

    # Unique
    unq_out = op_unique.infer_shape(t2, axis=0)
    assert isinstance(unq_out[0], SymVar)
    assert unq_out[1] == 8


def test_slicing_shapes() -> None:
    """Verify IndexInDim and UpdateSlice shape inference."""
    op_idx = IndexInDim()
    op_update = UpdateSlice()

    t3 = _make_tensor((2, 4, 8))
    idx = _make_tensor((3,))

    # IndexInDim with keepdims
    assert op_idx.infer_shape(t3, idx, axis=1, keepdims=True) == (2, 3, 8)
    assert op_idx.infer_shape(t3, idx, axis=1, keepdims=False) == (2, 3, 8)

    # UpdateSlice preserves operand shape
    assert op_update.infer_shape(t3) == (2, 4, 8)


def test_creation_frontend_utils_shapes() -> None:
    """Verify Geometric, Geomspace, FromDlpack, and Frompyfunc shape inference."""
    op_geom = Geometric()
    op_gs = Geomspace()
    op_dlpack = FromDlpack()
    op_pyfunc = Frompyfunc()

    p = _make_tensor((4, 5))

    # Geometric with explicit size and default size
    assert op_geom.infer_shape(p, size=(10, 20)) == (10, 20)
    assert op_geom.infer_shape(p) == (4, 5)

    # Geomspace
    start = _make_tensor((2, 3))
    stop = _make_tensor((2, 3))
    assert op_gs.infer_shape(start, stop, num=10, axis=0) == (10, 2, 3)

    # FromDlpack
    assert op_dlpack.infer_shape(p) == (4, 5)

    # Frompyfunc
    t1 = _make_tensor((2, 1))
    t2 = _make_tensor((1, 4))
    assert op_pyfunc.infer_shape(None, 2, 1, t1, t2) == (2, 4)


def test_nn_activations_shapes() -> None:
    """Verify HardSilu, HardSwish, and Squareplus shape inference."""
    op_hsilu = HardSilu()
    op_hswish = HardSwish()
    op_sqp = Squareplus()

    t = _make_tensor((3, 7, 11))

    assert op_hsilu.infer_shape(t) == (3, 7, 11)
    assert op_hswish.infer_shape(t) == (3, 7, 11)
    assert op_sqp.infer_shape(t) == (3, 7, 11)


def test_distributed_ops_shapes() -> None:
    """Verify Outfeed, Pshuffle, Pswapaxes, Ppermute, PsumScatter, Pmax, Pmin shape inference."""
    op_outfeed = Outfeed()
    op_shuffle = Pshuffle()
    op_swap = Pswapaxes()
    op_perm = Ppermute()
    op_scatter = PsumScatter()
    op_pmax = Pmax()
    op_pmin = Pmin()

    t = _make_tensor((4, 8, 16))

    assert op_outfeed.infer_shape(t) == ()
    assert op_shuffle.infer_shape(t) == (4, 8, 16)
    assert op_perm.infer_shape(t) == (4, 8, 16)
    assert op_pmax.infer_shape(t) == (4, 8, 16)
    assert op_pmin.infer_shape(t) == (4, 8, 16)

    # Pswapaxes
    assert op_swap.infer_shape(t, axis1=0, axis2=2) == (16, 8, 4)

    # PsumScatter: (4, 8, 16) with world_size=2 along dim 1 -> (4, 4, 16)
    assert op_scatter.infer_shape(t, scatter_dimension=1, world_size=2) == (4, 4, 16)


def test_unary_logical_shapes() -> None:
    """Verify Iscomplex, Iscomplexobj, Isreal, Isrealobj, Issubdtype, Isin, and Ediff1d shape inference."""
    op_c = Iscomplex()
    op_c_obj = Iscomplexobj()
    op_r = Isreal()
    op_r_obj = Isrealobj()
    op_sub = Issubdtype()
    op_isin = Isin()
    op_ediff = Ediff1d()

    t = _make_tensor((4, 10))

    assert op_c.infer_shape(t) == (4, 10)
    assert op_r.infer_shape(t) == (4, 10)
    assert op_c_obj.infer_shape(t) == ()
    assert op_r_obj.infer_shape(t) == ()
    assert op_sub.infer_shape(int, float) == ()

    test_elts = _make_tensor((5,))
    assert op_isin.infer_shape(t, test_elts) == (4, 10)

    # Ediff1d: 40 elements flattened -> 39 elements
    assert op_ediff.infer_shape(t) == (39,)


def test_foreign_call_shape_inference() -> None:
    """Verify ForeignCall shape inference using AST reflection and signatures."""
    op_foreign = ForeignCall()

    # Function returning constant shape tuple via AST reflection
    assert op_foreign.infer_shape(_sample_ast_reflected_func) == (10, 20)

    # Fallback to second argument tensor shape
    t = _make_tensor((5, 15))
    assert op_foreign.infer_shape(lambda x: x, t) == (5, 15)
