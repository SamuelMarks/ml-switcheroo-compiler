import ml_switcheroo_compiler.ops.shape.misc as sm


def test_shape_misc_coverage():
    ops = [
        "Nonzero",
        "Percentile",
        "Ppermute",
        "PsumScatter",
        "Quantile",
        "RavelMultiIndex",
        "Repeat",
        "Searchsorted",
        "Tile",
        "Unique",
        "UpdateSlice",
        "Flatnonzero",
        "IndexInDim",
        "Lexsort",
        "SortComplex",
        "Compress",
        "FillDiagonal",
        "Intersect1d",
        "Put",
        "Setdiff1d",
        "Setxor1d",
        "Union1d",
        "Extract",
        "NumpyIsneginf",
        "NumpyIsposinf",
        "TrimZeros",
    ]

    for op_name in ops:
        if hasattr(sm, op_name):
            cls = getattr(sm, op_name)
            inst = cls()
            inst.infer_shape()

            class DummyShape:
                shape = (1,)

            try:
                inst.infer_shape(DummyShape())
            except:
                pass
            try:
                inst.infer_shape(DummyShape(), DummyShape())
            except:
                pass
