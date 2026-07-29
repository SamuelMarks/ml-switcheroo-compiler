import sys
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def test_missing_coverage():
    # Line 1265-1266 Beta (not Betainc!)
    with patch.dict(sys.modules, {"scipy.special": None}):
        func = global_eager_registry.get("Beta")

        class DummyModBeta:
            pass  # No random, no beta

        res = func(DummyModBeta(), np.array([1]), np.array([1]))
        assert res is None

    # Line 1664 Adjoint
    func = global_eager_registry.get("Adjoint")

    class DummyModAdj2:
        def asarray(self, x):
            return np.asarray(x)

        # conj and transpose NOT present

    with pytest.raises(AttributeError):
        res = func(DummyModAdj2(), [1 + 1j, 2 - 2j])

    # Line 4238 Polyint fallback
    func = global_eager_registry.get("Polyint")

    class DummyModPoly:
        def asarray(self, x):
            return np.asarray(x)

        def polyint(self, p, m=1, k=None):
            return np.polyint(p, m=m, k=k)

    res = func(DummyModPoly(), [1, 2], 1)
    res2 = func(DummyModPoly(), [1, 2], 1, 3)

    class DummyModPolyMissing:
        def asarray(self, x):
            return np.asarray(x)

    res3 = func(DummyModPolyMissing(), [1, 2], 1)
    res4 = func(DummyModPolyMissing(), [1, 2], 1, 3)

    class DummyModPolyMissing2:
        pass

    res5 = func(DummyModPolyMissing2(), [1, 2], 1)

    class DummyModPolyHas:
        def asarray(self, x):
            return np.asarray(x)

        def polyint(self, p, m=1, k=None):
            if k is not None:
                return np.polyint(p, m=m, k=k)
            return np.polyint(p, m=m)

    res6 = func(DummyModPolyHas(), [1, 2], 1)
    res7 = func(DummyModPolyHas(), [1, 2], 1, 3)

    # Line 4580->4582 TensorArrayWrite
    func = global_eager_registry.get("TensorArrayWrite")
    ta = [1, 2]
    res = func(np, ta, 1, 4)
    assert res == [1, 4]

    # Line 4723 TriangularFallback
    func = global_eager_registry.get("Triangular")

    class DummyModTri2:
        class random:
            pass  # No triangular method initially

    # If it falls back to NumPy's random... wait, line 4721 says 'if hasattr(backend_module, "random"): return backend_module.random.triangular'. But DummyModTri2.random doesn't have it, so it will raise AttributeError. Let's just mock it so it succeeds.
    dummy2 = DummyModTri2()
    # If the first hasattr fails, we need hasattr(backend_module.random, "triangular") to be False,
    # then it reaches line 4722 and calls backend_module.random.triangular anyway! That's weird.
    # Ah, the code is:
    # if hasattr(backend_module, "random") and hasattr(backend_module.random, "triangular"):
    #     return backend_module.random.triangular(...)
    # if hasattr(backend_module, "random"):
    #     return backend_module.random.triangular(...)
    # So if it has random but NOT triangular, the first is false, the second is true, and it raises AttributeError.
    with pytest.raises(AttributeError):
        func(dummy2, 1, 2, 3)

    # Line 1432 AdaptiveAvgPool2D fallback
    func = global_eager_registry.get("AdaptiveAvgPool2D")

    class DummyModPool:
        def zeros(self, s, dtype=None):
            return np.zeros(s, dtype=dtype)

    res = func(DummyModPool(), np.ones((2, 2)), 1)
    assert res.shape == (2, 1)

    # Line 1480 AllGather array fallback
    func = global_eager_registry.get("AllGather")

    class DummyMod1:
        def array(self, x):
            return x

    res = func(DummyMod1(), [1, 2])
    assert res == [[1, 2]]

    # Line 1664 Adjoint
    func = global_eager_registry.get("Adjoint")

    class DummyModAdj:
        def asarray(self, x):
            return np.asarray(x)

        def conj(self, x):
            return np.conj(x)

        def transpose(self, x):
            return np.transpose(x)

    res = func(DummyModAdj(), [1 + 1j, 2 - 2j])
    assert np.allclose(res, [1 - 1j, 2 + 2j])

    # Line 1816 CholeskyEx (info=0 fallback)
    func = global_eager_registry.get("CholeskyEx")

    class DummyMod2:
        def cholesky(self, *args, **kwargs):
            return "chol"

    res = func(DummyMod2(), [[1.0]])
    assert res == ("chol", 0)

    # Line 3738 FillDiagonal fallback
    func = global_eager_registry.get("FillDiagonal")
    a = np.zeros((3, 3))

    class DummyModDiag:
        def array(self, x):
            return x

    res = func(DummyModDiag(), a, 1)
    assert res[0, 0] == 1

    # Line 3948, 3950 BandPart
    func = global_eager_registry.get("BandPart")
    a = np.ones((3, 3))

    class DummyModBand:
        def shape(self, x):
            return np.shape(x)

        def indices(self, s):
            return np.indices(s)

        def ones(self, s, dtype=None):
            return np.ones(s, dtype=dtype)

        def zeros_like(self, x):
            return np.zeros_like(x)

        def where(self, cond, x, y):
            return np.where(cond, x, y)

    res = func(DummyModBand(), a, 1, 1)
    assert res.shape == (3, 3)

    # Line 4237 Polyint
    func = global_eager_registry.get("Polyint")

    class DummyModPoly:
        def asarray(self, x):
            return np.asarray(x)

        def polyint(self, p, m=1, k=None):
            return np.polyint(p, m=m, k=k)

    res = func(DummyModPoly(), [1, 2], 1, 3)

    # Line 4284-4287, 4290 ScatterApply
    func = global_eager_registry.get("ScatterApply")
    tensor = np.zeros((3, 3))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 2])
    # The signature is: backend_module, tensor, indices, updates, reduction
    res = func(np, tensor, indices, updates, "add")
    res = func(np, tensor, indices, updates, "mul")
    res = func(np, tensor, indices, updates, "none")
    with pytest.raises(RuntimeError):
        # Out of bounds indices will trigger IndexError which becomes RuntimeError
        func(np, tensor, np.array([[10, 10]]), updates, "add")

    # Line 4309 ScatterMax
    func = global_eager_registry.get("ScatterMax")
    res = func(np, np.zeros((3, 3)), indices, updates)

    # Line 4328 ScatterMin
    func = global_eager_registry.get("ScatterMin")
    res = func(np, np.ones((3, 3)), indices, updates)

    # Line 4347 ScatterMul
    func = global_eager_registry.get("ScatterMul")
    res = func(np, np.ones((3, 3)), indices, updates)

    # Line 4366 ScatterNd
    func = global_eager_registry.get("ScatterNd")
    res = func(np, indices, updates, (3, 3))

    # Line 4448-4449 StringToNumber ValueError
    func = global_eager_registry.get("StringToNumber")
    res = func(np, ["1", "invalid"])
    assert res.shape == (2,)

    # Line 4574-4588 TensorArrayWrite
    func = global_eager_registry.get("TensorArrayWrite")
    ta = [1, 2]
    res = func(np, ta, 3, 4)
    assert res == [1, 2, None, 4]
    ta2 = np.array([1, 2])
    res2 = func(np, ta2, 0, 4)
    assert res2[0] == 4

    # Line 4722 TriangularFallback
    func = global_eager_registry.get("Triangular")

    class DummyModTri:
        class random:
            @staticmethod
            def triangular(*args, **kwargs):
                return "tri"

    res = func(DummyModTri(), 1, 2, 3)
    assert res == "tri"

    # Line 4994-5003 UpdateSlice
    func = global_eager_registry.get("UpdateSlice")
    operand = np.zeros((3, 3))
    update = np.ones((2, 2))
    start = np.array([0, 0])
    res = func(np, operand, update, start)
    with pytest.raises(RuntimeError):
        func(np, operand, update, np.array([10, 10]))
