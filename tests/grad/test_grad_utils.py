import pytest

from ml_switcheroo_compiler.grad.utils import _get_inputs_dict, _to_original_type


def test_get_inputs_dict_missing_input(mocker):
    """Test function."""

    class DummyGraph:
        inputs = ["missing_inp"]
        nodes = {}

    with pytest.raises(ValueError, match="Missing input value for node 'missing_inp'"):
        _get_inputs_dict(DummyGraph())


def test_to_original_type_dtypes(mocker):
    """Test function."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig

    orig = Tensor(None, TensorConfig((), DType.Float32, None))

    class DummyBackend:
        def asarray(self, val):
            return val

    mocker.patch("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=DummyBackend())

    val_int = np.array(1, dtype=np.int32)
    res_int = _to_original_type(val_int, orig)
    assert res_int.dtype == DType.Int32

    val_bool = np.array(True, dtype=bool)
    res_bool = _to_original_type(val_bool, orig)
    assert res_bool.dtype == DType.Bool


def test_find_wrt_tensors(mocker):
    """Test function."""
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.utils import _find_wrt_tensors

    class DummyData:
        id = "n"

    class DummyGraph:
        nodes = {"n": {}}

    class DummyTensor1(Tensor):
        @property
        def requires_grad(self):
            return True

        @property
        def trainable(self):
            return False

    class DummyTensor2(Tensor):
        @property
        def requires_grad(self):
            return False

        @property
        def trainable(self):
            return True

    t = DummyTensor1(None, TensorConfig((), DType.Float32, None))
    t._data = DummyData()
    t2 = DummyTensor2(None, TensorConfig((), DType.Float32, None))
    t2._data = DummyData()

    wrt_tensors, wrt_ids = _find_wrt_tensors(DummyGraph())
    assert id(t) in [id(x) for x in wrt_tensors]
    assert id(t2) in [id(x) for x in wrt_tensors]
    assert "n" in wrt_ids
