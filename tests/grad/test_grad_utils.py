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

    class DummyTensor3(Tensor):
        @property
        def requires_grad(self):
            return True

    class DummyDataNotInGraph:
        id = "not_in_graph"

    t3 = DummyTensor3(None, TensorConfig((), DType.Float32, None))
    t3._data = DummyDataNotInGraph()

    wrt_tensors, wrt_ids = _find_wrt_tensors(DummyGraph())
    assert id(t) in [id(x) for x in wrt_tensors]
    assert id(t2) in [id(x) for x in wrt_tensors]
    assert id(t3) not in [id(x) for x in wrt_tensors]
    assert "n" in wrt_ids


def test_get_concrete_val_variants():
    """Test _get_concrete_val with non-proxy and proxy objects."""
    from ml_switcheroo_compiler.grad.utils import _get_concrete_val

    class NonProxy:
        _data = 42

    assert _get_concrete_val(NonProxy()) == 42


def test_get_inputs_dict_success(mocker):
    """Test _get_inputs_dict full branch coverage including successful inputs lookup."""
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    class DummyBackend:
        def asarray(self, val):
            return val

    mocker.patch("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=DummyBackend())

    class NodeData:
        def __init__(self, id, concrete_value=None):
            self.id = id
            self.concrete_value = concrete_value

    class DummyGraphSuccess:
        inputs = ["in_valid"]
        nodes = {"in_valid": {}, "no_val_node": {}}

    t_valid = Tensor(None, TensorConfig((), DType.Float32, None))
    t_valid._data = NodeData("in_valid", concrete_value=123.0)

    t_no_val = Tensor(None, TensorConfig((), DType.Float32, None))
    proxy = ProxyTensor(None, ())
    proxy.id = "no_val_node"
    proxy.concrete_value = None
    t_no_val._data = proxy

    t_not_in_graph = Tensor(None, TensorConfig((), DType.Float32, None))
    t_not_in_graph._data = NodeData("orphan_node", concrete_value=999.0)

    t_no_id = Tensor(1.0, TensorConfig((), DType.Float32, None))

    res = _get_inputs_dict(DummyGraphSuccess())
    assert "in_valid" in res
    assert res["in_valid"] == 123.0
    assert "no_val_node" not in res
    assert "orphan_node" not in res

    class GraphNoInputs:
        nodes = {}

    res_no_inputs = _get_inputs_dict(GraphNoInputs())
    assert isinstance(res_no_inputs, dict)
