def test_shape_misc_stubs():
    from ml_switcheroo_compiler.ops.shape.misc import Flatnonzero, Lexsort, Nonzero, Percentile, Quantile, RavelMultiIndex, Repeat, Searchsorted, SortComplex, Tile, Unique

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Flatnonzero, Lexsort, Nonzero, Percentile, Quantile, RavelMultiIndex, Repeat, Searchsorted, SortComplex, Tile, Unique]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        assert op.infer_shape(Dummy(), Dummy()) is not None
        assert op.infer_shape() is not None


def test_shape_slicing_stubs():
    from ml_switcheroo_compiler.ops.shape.slicing import IndexInDim, UpdateSlice

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [IndexInDim, UpdateSlice]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        try:
            op.infer_shape()
        except Exception:
            pass


def test_creation_frontend_stubs():
    from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Geometric, Geomspace, FromDlpack, Frompyfunc]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        assert op.infer_shape(Dummy(), Dummy(), Dummy()) is not None
        assert op.infer_shape(size=(1, 2)) is not None
        assert op.infer_shape(num=5) is not None
        assert op.infer_shape() is not None


def test_nn_activations_stubs():
    from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, Squareplus

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [HardSilu, HardSwish, Squareplus]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        try:
            op.infer_shape()
        except Exception:
            pass


def test_distributed_stubs():
    from ml_switcheroo_compiler.ops.distributed_ops import Outfeed, Pmax, Pmin, Ppermute, Pshuffle, PsumScatter, Pswapaxes

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Outfeed, Pshuffle, Pswapaxes, Ppermute, PsumScatter, Pmax, Pmin]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        try:
            op.infer_shape()
        except Exception:
            pass


def test_random_stubs():
    from ml_switcheroo_compiler.ops.random_ops.core import Rademacher

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Rademacher]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        assert op.infer_shape(shape=(1, 2)) is not None
        assert op.infer_shape() is not None


def test_unary_logical_stubs():
    from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplex, Iscomplexobj, Isin, Isreal, Isrealobj, Issubdtype

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Iscomplex, Iscomplexobj, Isreal, Isrealobj, Issubdtype, Isin, Ediff1d]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        try:
            op.infer_shape()
        except Exception:
            pass


def test_linalg_dot_stubs():
    from ml_switcheroo_compiler.ops.linalg.dot import Pdot

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Pdot]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        try:
            op.infer_shape()
        except Exception:
            pass


def test_io_stubs():
    from ml_switcheroo_compiler.ops.io import Fromfile, Fromfunction, Fromiter, Fromstring

    class Dummy:
        shape = (2, 2)

        def __iter__(self):
            return iter(self.shape)

    for OpClass in [Fromfile, Fromstring, Fromiter, Fromfunction]:
        op = OpClass()
        assert op.infer_shape(Dummy()) is not None
        assert op.infer_shape(Dummy(), Dummy()) is not None
        assert op.infer_shape(shape=(2, 2)) is not None
        assert op.infer_shape() is not None
