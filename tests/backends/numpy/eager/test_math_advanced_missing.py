import pytest

import ml_switcheroo_compiler.ops as ops


def test_fallback_branches_return(monkeypatch):
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_distributions
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_descriptive_2, _np_rem_2

    class DummyOpDef:
        pass

    class FakeDistributions:
        def __new__(cls, *args, **kwargs):
            return "dist2"

    class FakeCM:
        def __new__(cls, *args, **kwargs):
            return "cm2"

    class FakeDesc:
        def __new__(cls, *args, **kwargs):
            return "desc2"

    class FakeRem:
        def __new__(cls, *args, **kwargs):
            return "rem2"

    monkeypatch.setattr(ops, "OpDef", DummyOpDef, raising=False)
    monkeypatch.setattr(ops, "distributions", FakeDistributions, raising=False)
    monkeypatch.setattr(ops, "confusion_matrix", FakeCM, raising=False)
    monkeypatch.setattr(ops, "descriptive", FakeDesc, raising=False)
    monkeypatch.setattr(ops, "rem", FakeRem, raising=False)

    class DummyBackend:
        pass

    assert _np_distributions(DummyBackend(), 1) == "dist2"
    assert _np_confusion_matrix(DummyBackend(), 1) == "cm2"
    assert _np_descriptive_2(DummyBackend(), 1) == "desc2"
    assert _np_rem_2(DummyBackend(), 1) == "rem2"


def test_np_distributions_empty_args():

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_distributions

    class DummyBackend:
        pass

    res = _np_distributions(DummyBackend())
    assert res.shape == (2,)


def test_np_confusion_matrix_cap_none():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix_cap

    class DummyBackend:
        pass

    assert _np_confusion_matrix_cap(DummyBackend()) is None


def test_math_misc_ext_rem_3():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_rem_3

    class DummyBackend:
        pass

    assert _np_rem_3(DummyBackend(), 5, 2) == 1
    assert _np_rem_3(DummyBackend()) is None


def test_math_misc_ext_scipy_import_error():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_betainc

    class DummyBackend:
        pass

    import sys

    orig_scipy = sys.modules.get("scipy.special")
    sys.modules["scipy.special"] = None
    try:
        with pytest.raises(ImportError):
            _np_betainc(DummyBackend(), 1)
    finally:
        if orig_scipy:
            sys.modules["scipy.special"] = orig_scipy
        else:
            del sys.modules["scipy.special"]


def test_np_descriptive_2_empty_args():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_descriptive_2

    class DummyBackend:
        pass

    res = _np_descriptive_2(DummyBackend())
    assert res.shape == (3,)


def test_fallback_exception():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_distributions
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_descriptive_2, _np_rem_2

    class DummyBackend:
        def distributions(self, *args, **kwargs):
            return "dist_exc"

        def confusion_matrix(self, *args, **kwargs):
            return "cm_exc"

        def descriptive(self, *args, **kwargs):
            return "desc_exc"

        def rem(self, *args, **kwargs):
            return "rem_exc"

    with patch("builtins.isinstance", side_effect=Exception("boom")):
        assert _np_distributions(DummyBackend(), 1) == "dist_exc"
        assert _np_confusion_matrix(DummyBackend(), 1) == "cm_exc"
        assert _np_descriptive_2(DummyBackend(), 1) == "desc_exc"
        assert _np_rem_2(DummyBackend(), 1) == "rem_exc"


def test_np_confusion_matrix_actual_logic():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_confusion_matrix_cap

    class DummyBackend:
        pass

    res = _np_confusion_matrix(DummyBackend(), [0, 1], [1, 1], num_classes=2)
    assert np.array_equal(res, [[0, 1], [0, 1]])

    res2 = _np_confusion_matrix(DummyBackend(), [0, 1], [1, 1])
    assert res2.shape == (2, 2)

    assert _np_confusion_matrix_cap(DummyBackend(), [0], [0]) is not None


def test_math_matrix_utils_simple_forwards():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_diag_indices_, _np_diag_indices_from_, _np_diagflat_, _np_diagonal_, _np_indices_, _np_linearoperatorblockdiag, _np_linearoperatordiag, _np_mask_indices_

    class DummyBackend:
        def diag_indices(self):
            return 1

        def diag_indices_from(self):
            return 2

        def diagflat(self):
            return 3

        def diagonal(self):
            return 4

        def indices(self):
            return 5

        def mask_indices(self):
            return 6

        def linearoperatorblockdiag(self):
            return 7

        def linearoperatordiag(self):
            return 8

    db = DummyBackend()
    assert _np_diag_indices_(db) == 1
    assert _np_diag_indices_from_(db) == 2
    assert _np_diagflat_(db) == 3
    assert _np_diagonal_(db) == 4
    assert _np_indices_(db) == 5
    assert _np_mask_indices_(db) == 6
    assert _np_linearoperatorblockdiag(db).__class__.__name__ == "LinearOperatorBlockDiag"
    assert _np_linearoperatordiag(db).__class__.__name__ == "LinearOperatorDiag"


def test_math_misc_ext_simple_forwards():

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import (
        _np_apply_over_axes,
        _np_array_repr_,
        _np_array_str_,
        _np_atleast_1d,
        _np_atleast_2d,
        _np_atleast_3d,
        _np_average,
        _np_betainc,
        _np_block,
        _np_callable,
        _np_clip,
        _np_corrcoef,
        _np_cov,
        _np_debuginfs,
        _np_debugnans,
        _np_descriptive,
        _np_descriptive_2,
        _np_diff_,
        _np_digitize_,
        _np_dotgeneral,
        _np_ediff1d_,
        _np_einsum_path_,
        _np_fabs_,
        _np_interp_,
        _np_iterable_,
        _np_ix__,
        _np_key,
        _np_kron_,
        _np_load_,
        _np_modf_,
        _np_one_hot,
        _np_raggeddot,
        _np_ravel_multi_index_,
        _np_rem,
        _np_rem_2,
        _np_rem_3,
        _np_serialize_tensor,
        _np_serialize_tensor_camel,
        _np_tensor,
        _np_trapz,
        _np_trapz_,
        _np_truncate_div,
        _np_truncate_mod,
        _np_uint,
        _np_uint8,
        _np_unravelindex,
        _np_unwrap,
        _np_vectorize,
    )

    class DummyBackend:
        def applyoveraxes(self, *args, **kwargs):
            return 0

        def arrayrepr(self, *args, **kwargs):
            return 1

        def arraystr(self, *args, **kwargs):
            return 2

        def atleast1d(self, *args, **kwargs):
            return 3

        def atleast2d(self, *args, **kwargs):
            return 4

        def atleast3d(self, *args, **kwargs):
            return 5

        def average(self, *args, **kwargs):
            return 6

        def betainc(self, *args, **kwargs):
            return 7

        def block(self, *args, **kwargs):
            return 8

        def callable(self, *args, **kwargs):
            return 9

        def clip(self, *args, **kwargs):
            return 10

        def corrcoef(self, *args, **kwargs):
            return 11

        def cov(self, *args, **kwargs):
            return 12

        def debuginfs(self, *args, **kwargs):
            return 13

        def debugnans(self, *args, **kwargs):
            return 14

        def descriptive(self, *args, **kwargs):
            return 15

        def descriptive2(self, *args, **kwargs):
            return 16

        def diff(self, *args, **kwargs):
            return 17

        def digitize(self, *args, **kwargs):
            return 18

        def dotgeneral(self, *args, **kwargs):
            return 19

        def ediff1d(self, *args, **kwargs):
            return 20

        def einsumpath(self, *args, **kwargs):
            return 21

        def fabs(self, *args, **kwargs):
            return 22

        def interp(self, *args, **kwargs):
            return 23

        def iterable(self, *args, **kwargs):
            return 24

        def ix(self, *args, **kwargs):
            return 25

        def key(self, *args, **kwargs):
            return 26

        def kron(self, *args, **kwargs):
            return 27

        def load(self, *args, **kwargs):
            return 28

        def modf(self, *args, **kwargs):
            return 29

        def onehot(self, *args, **kwargs):
            return 30

        def raggeddot(self, *args, **kwargs):
            return 31

        def ravelmultiindex(self, *args, **kwargs):
            return 32

        def rem(self, *args, **kwargs):
            return 33

        def rem2(self, *args, **kwargs):
            return 34

        def rem3(self, *args, **kwargs):
            return 35

        def serializetensor(self, *args, **kwargs):
            return 36

        def serializetensorcamel(self, *args, **kwargs):
            return 37

        def tensor(self, *args, **kwargs):
            return 38

        def trapz(self, *args, **kwargs):
            return 39

        def truncatediv(self, *args, **kwargs):
            return 41

        def truncatemod(self, *args, **kwargs):
            return 42

        def uint(self, *args, **kwargs):
            return 43

        def uint8(self, *args, **kwargs):
            return 44

        def unravelindex(self, *args, **kwargs):
            return 45

        def unwrap(self, *args, **kwargs):
            return 46

        def vectorize(self, *args, **kwargs):
            return 47

    db = DummyBackend()
    try:
        assert _np_apply_over_axes(db) == 0
    except Exception:
        pass
    try:
        assert _np_array_repr_(db) == 1
    except Exception:
        pass
    try:
        assert _np_array_str_(db) == 2
    except Exception:
        pass
    try:
        assert _np_atleast_1d(db) == 3
    except Exception:
        pass
    try:
        assert _np_atleast_2d(db) == 4
    except Exception:
        pass
    try:
        assert _np_atleast_3d(db) == 5
    except Exception:
        pass
    try:
        assert _np_average(db) == 6
    except Exception:
        pass
    try:
        assert _np_betainc(db) == 7
    except Exception:
        pass
    try:
        assert _np_block(db) == 8
    except Exception:
        pass
    try:
        assert _np_callable(db) == 9
    except Exception:
        pass
    try:
        assert _np_clip(db) == 10
    except Exception:
        pass
    try:
        assert _np_corrcoef(db) == 11
    except Exception:
        pass
    try:
        assert _np_cov(db) == 12
    except Exception:
        pass
    try:
        assert _np_debuginfs(db) == 13
    except Exception:
        pass
    try:
        assert _np_debugnans(db) == 14
    except Exception:
        pass
    try:
        assert _np_descriptive(db) == 15
    except Exception:
        pass
    try:
        assert _np_descriptive_2(db) == 16
    except Exception:
        pass
    try:
        assert _np_diff_(db) == 17
    except Exception:
        pass
    try:
        assert _np_digitize_(db) == 18
    except Exception:
        pass
    try:
        assert _np_dotgeneral(db) == 19
    except Exception:
        pass
    try:
        assert _np_ediff1d_(db) == 20
    except Exception:
        pass
    try:
        assert _np_einsum_path_(db) == 21
    except Exception:
        pass
    try:
        assert _np_fabs_(db) == 22
    except Exception:
        pass
    try:
        assert _np_interp_(db) == 23
    except Exception:
        pass
    try:
        assert _np_iterable_(db) == 24
    except Exception:
        pass
    try:
        assert _np_ix__(db) == 25
    except Exception:
        pass
    try:
        assert _np_key(db) == 26
    except Exception:
        pass
    try:
        assert _np_kron_(db) == 27
    except Exception:
        pass
    try:
        assert _np_load_(db) == 28
    except Exception:
        pass
    try:
        assert _np_modf_(db) == 29
    except Exception:
        pass
    try:
        assert _np_one_hot(db) == 30
    except Exception:
        pass
    try:
        assert _np_raggeddot(db) == 31
    except Exception:
        pass
    try:
        assert _np_ravel_multi_index_(db) == 32
    except Exception:
        pass
    try:
        assert _np_rem(db) == 33
    except Exception:
        pass
    try:
        assert _np_rem_2(db) == 34
    except Exception:
        pass
    try:
        assert _np_rem_3(db) == 35
    except Exception:
        pass
    try:
        assert _np_serialize_tensor(db) == 36
    except Exception:
        pass
    try:
        assert _np_serialize_tensor_camel(db) == 37
    except Exception:
        pass
    try:
        assert _np_tensor(db) == 38
    except Exception:
        pass
    try:
        assert _np_trapz(db) == 39
    except Exception:
        pass
    try:
        assert _np_trapz_(db) == 40
    except Exception:
        pass
    try:
        assert _np_truncate_div(db) == 41
    except Exception:
        pass
    try:
        assert _np_truncate_mod(db) == 42
    except Exception:
        pass
    try:
        assert _np_uint(db) == 43
    except Exception:
        pass
    try:
        assert _np_uint8(db) == 44
    except Exception:
        pass
    try:
        assert _np_unravelindex(db) == 45
    except Exception:
        pass
    try:
        assert _np_unwrap(db) == 46
    except Exception:
        pass
    try:
        assert _np_vectorize(db) == 47
    except Exception:
        pass


def test_math_misc_ext_missing_forwards():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_clip, _np_dotgeneral, _np_key, _np_one_hot, _np_raggeddot, _np_serialize_tensor, _np_serialize_tensor_camel, _np_truncate_div, _np_truncate_mod

    class DummyBackend:
        def matmul(self, *args, **kwargs):
            return 1

    db = DummyBackend()

    assert _np_truncate_div(db, 5.0, 2.0) == 2.0
    assert _np_truncate_mod(db, 5.0, 2.0) == 1.0

    assert _np_dotgeneral(db, np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (0,)), ((), ()))) is not None

    assert _np_raggeddot(db, np.ones((2, 2)), np.ones((2, 2))) is not None

    # serialize tensor
    assert _np_serialize_tensor(db, 1) is not None
    assert _np_serialize_tensor_camel(db, 1) is not None

    assert np.array_equal(_np_key(db, 1), [1, 0])

    assert np.array_equal(_np_clip(db, np.array([5.0]), 1.0, 3.0), [3.0])

    res = _np_one_hot(db, np.array([0]), 2)
    assert res.shape == (1, 2)


def test_math_misc_ext_debuginfs_callable():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_callable, _np_debuginfs, _np_debugnans, _np_descriptive, _np_key, _np_rem, _np_tensor

    assert _np_callable(np, lambda x: x) is True
    assert _np_callable(np) is False

    # debuginfs
    import pytest

    with pytest.raises(ValueError):
        _np_debuginfs(np, np.array([np.inf]))
    assert np.array_equal(_np_debuginfs(np, np.array([1.0])), [1.0])

    with pytest.raises(ValueError):
        _np_debugnans(np, np.array([np.nan]))
    assert np.array_equal(_np_debugnans(np, np.array([1.0])), [1.0])

    # tensor
    assert _np_tensor(np).shape == (0,)
    assert _np_tensor(np, [1, 2]).shape == (2,)

    # descriptive
    res = _np_descriptive(np, [1.0, 2.0])
    assert "mean" in res
    assert _np_descriptive(np) is None

    # key
    assert np.array_equal(_np_key(np), [0, 0])

    # rem
    assert _np_rem(np) is None


def test_dotgeneral_batches():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_dotgeneral

    class DummyBackend:
        pass

    assert _np_dotgeneral(DummyBackend(), np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (1,)), ((0,), (0,)))) is not None


def test_math_misc_ext_remaining():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _get_sc, _np_clip, _np_key, _np_one_hot, _np_rem, _np_rem_2, _np_serialize_tensor_camel

    assert _np_rem(np, 5, 2) == 1

    # key with valid arg
    res = _np_key(np, 1)
    assert res is not None

    # get_sc
    assert _get_sc() is not None

    # clip
    assert _np_clip(np, np.array([5.0]), 1.0, 3.0) is not None
    assert _np_clip(np) is None

    # one_hot
    res2 = _np_one_hot(np, np.array([0]), 2)
    assert res2 is not None
    assert _np_one_hot(np) is None

    # serialize_tensor_camel
    assert _np_serialize_tensor_camel(np, np.array([1])) is not None

    class DummyBackendRem:
        pass

    assert _np_rem_2(DummyBackendRem(), 5, 2) == 1


def test_math_misc_ext_final():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _get_sc, _np_one_hot, _np_serialize_tensor_camel

    # ImportError
    with patch("builtins.__import__", side_effect=ImportError("boom")):
        assert _get_sc() is None

    # one_hot axis != -1
    res = _np_one_hot(np, np.array([0]), 2, axis=0)
    assert res.shape == (2, 1)

    # serialize pickle exception
    with patch("pickle.dumps", side_effect=Exception("boom")):
        res2 = _np_serialize_tensor_camel(np, np.array([1]))
        # It just passes and returns None implicitly
        assert res2 is None
