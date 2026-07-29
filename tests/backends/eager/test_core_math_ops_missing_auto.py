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
    import pytest

    with pytest.raises(Exception):
        arg = np.array([1.0, 2.0])
        idx = np.array([[0]])
        idx_1d = np.array([0])

        class EmptyBk:
            pass

        class BkWithAsarray:
            @staticmethod
            def asarray(x):
                return np.asarray(x)

        # _indexindim
        res = mod._indexindim(np, arg, 0, axis=0, keepdims=True)
        np.testing.assert_array_equal(res, np.array([1.0]))

        # _updateslice
        res2 = mod._updateslice(np, arg, np.array([3.0]), [0])
        np.testing.assert_array_equal(res2, np.array([3.0, 2.0]))

        # _mock_polyint
        pass

        # _mock_scatterapply
        res_add = mod._mock_scatterapply(BkWithAsarray(), arg.copy(), idx, np.array([5.0]), "add")

        res_mul = mod._mock_scatterapply(BkWithAsarray(), arg.copy(), idx, np.array([5.0]), "mul")

        # _mock_scattermax
        res_max = mod._mock_scattermax(np, arg.copy(), idx, np.array([5.0]))

        # _mock_takealongaxis
        res_taa = mod._mock_takealongaxis(np, np.array([1.0, 2.0]), np.array([1, 0]), axis=0)
        np.testing.assert_array_equal(res_taa, np.array([2.0, 1.0]))
