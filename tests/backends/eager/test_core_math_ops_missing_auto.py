import numpy as np

import ml_switcheroo_compiler.backends.eager.core_math_ops as mod


def test_missing_mock_randoms():
    class DummyRandom:
        pass

    class BkWithRandom:
        def __init__(self):
            self.random = DummyRandom()

    bk = BkWithRandom()
    funcs_to_mock = [
        "stringsubstr",
        "stringtohash",
        "stringtonumber",
        "svdvals",
        "switch",
        "t",
        "takealongaxis",
        "tensorarrayread",
        "tensorarraystack",
        "tensorarraywrite",
        "tensorscattersub",
        "tensorscatterupdate",
        "textvectorization",
        "topk",
        "trapezoidalintegral",
        "triinv",
        "triangularsolve",
        "tridiagonalmatmul",
        "tridiagonalsolve",
        "unfold",
        "uniqueall",
        "uniquecounts",
        "uniqueinverse",
        "uniquevalues",
        "unstack",
        "variance",
        "vecdot",
        "vectornorm",
        "welch",
        "windowhamming",
        "windowhann",
        "writefile",
        "xdivy",
        "xlog1py",
        "xlogy",
    ]

    for f in funcs_to_mock:
        setattr(bk.random, f, lambda *a, **k: np.array(1.0))

    arg = np.array([1.0])

    for f in funcs_to_mock:
        mod_func = getattr(mod, f"_mock_{f}")
        mod_func(bk, arg)


def test_missing_others():
    arg = np.array([1.0, 2.0])
    idx = np.array([[0]])
    idx_1d = np.array([0])

    class EmptyBk:
        pass

    class BkWithAsarray:
        @staticmethod
        def asarray(x):
            return np.asarray(x)

    # _all_gather
    class BkWithArrayOnly:
        @staticmethod
        def array(x):
            return x

    mod._all_gather(BkWithArrayOnly(), arg)

    # _indexindim
    mod._indexindim(np, arg, 0, axis=0, keepdims=True)

    # _updateslice
    mod._updateslice(np, arg, np.array([3.0]), [0])

    # _mock_gcd
    # To hit AttributeError, we can temporarily remove np.gcd
    orig_gcd = getattr(np, "gcd", None)
    if orig_gcd:
        del np.gcd
    try:
        mod._mock_gcd(EmptyBk(), np.array([2]), np.array([4]))
    except Exception:
        pass
    if orig_gcd:
        np.gcd = orig_gcd

    # _mock_polyint
    mod._mock_polyint(EmptyBk(), np.array([1.0, 2.0]), m=1, k=0)

    # _mock_scatterapply
    mod._mock_scatterapply(BkWithAsarray(), arg, idx, np.array([5.0]), "add")
    mod._mock_scatterapply(BkWithAsarray(), arg, idx, np.array([5.0]), "mul")
    mod._mock_scatterapply(BkWithAsarray(), arg, idx, np.array([5.0]), "none")
    mod._mock_scatterapply(BkWithAsarray(), arg, np.array([[100]]), np.array([5.0]), "add")  # raises Exception

    # _mock_scattermax etc were tested already

    # _mock_triangular
    mod._mock_triangular(EmptyBk(), 0.0, 0.5, 1.0)

    # _mock_unfold
    mod._mock_unfold(BkWithAsarray(), arg)

    # _mock_updateslice
    mod._mock_updateslice(BkWithAsarray(), arg, np.array([3.0]), idx_1d)
    mod._mock_updateslice(BkWithAsarray(), arg, np.array([3.0, 4.0, 5.0]), np.array([0]))  # out of bounds

    # _mock_scattermax
    mod._mock_scattermax(np, arg, idx, np.array([5.0]))
    mod._mock_scattermin(np, arg, idx, np.array([5.0]))
    mod._mock_scattermul(np, arg, idx, np.array([5.0]))
    mod._mock_scatternd(np, idx, np.array([5.0]), (2,))

    # _mock_stringtonumber
    mod._mock_stringtonumber(np, np.array(["invalid_float"]))

    # _mock_takealongaxis
    mod._mock_takealongaxis(np, np.array([1.0, 2.0]), np.array([1, 0]), axis=0)

    orig_taa = getattr(np, "take_along_axis", None)
    if orig_taa:
        del np.take_along_axis
    try:
        mod._mock_takealongaxis(np, np.array([1.0, 2.0]), np.array([1, 0]))
    except Exception:
        pass
    if orig_taa:
        np.take_along_axis = orig_taa

    # _mock_tensorarraywrite
    mod._mock_tensorarraywrite(np, [], 2, arg)
    mod._mock_tensorarraywrite(np, [arg], 0, arg)
    mod._mock_tensorarraywrite(np, arg, 1, 5.0)
