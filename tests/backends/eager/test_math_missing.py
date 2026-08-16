import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite, _np_topk
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _all_gather, _np_updateslice, _updateslice
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_misc_ext import _adjoint, _indexindim


class DummyBackend:
    pass


def test_math_missing():
    # math_internal.py 79 (_np_tensorarraywrite)
    db = DummyBackend()
    arr = [0, 0, 0]
    res = _np_tensorarraywrite(db, arr, 1, 5)
    assert res == [0, 5, 0]

    # math_internal.py 99 (_np_topk)
    db = DummyBackend()
    arr = np.array([1, 3, 2, 4])
    vals, idx = _np_topk(db, arr, 2)
    assert list(vals) == [3, 4]
    assert list(idx) == [1, 3]

    # math_manipulation.py 26 (_all_gather)
    db = DummyBackend()  # no stack, no array
    tensor = 5
    res = _all_gather(db, tensor)
    assert res == 5

    # math_manipulation.py 156-157 (_updateslice)
    db = DummyBackend()
    db.array = np.array
    arr = np.zeros((3, 3))
    update = np.ones((2, 2))
    res = _updateslice(db, arr, update, [1, 1])
    assert res[1, 1] == 1.0

    # math_manipulation.py 367 (_np_updateslice)
    db = DummyBackend()
    arr = [0, 0]
    res = _np_updateslice(db, arr, 0, 5)
    assert res == [5, 0]

    # math_misc_ext.py 466 (_adjoint)
    db = DummyBackend()
    db.conj = lambda x: x  # dummy
    db.transpose = lambda x: x  # dummy
    db.asarray = lambda x: x

    class DbAdjoint:
        pass

    dba = DbAdjoint()
    dba.asarray = lambda x: x
    dba.transpose = lambda x: x
    # now it fails first if, goes to fallback
    dba.conj = lambda x: x  # re-add as instance attr
    res = _adjoint(dba, np.array([[1j]]))

    # math_misc_ext.py 736 (_indexindim keepdims=True)
    db = DummyBackend()
    db.array = np.array
    arr = np.array([1, 2, 3])
    res = _indexindim(db, arr, index=1, keepdims=True)
    assert list(res) == [2]
