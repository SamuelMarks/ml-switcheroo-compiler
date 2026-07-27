import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import _convert_to_tensors, _find_wrt_tensors, _generate_dummy_input, _get_concrete_val, _get_inputs_dict, _to_original_type


def test_grad_missing_branches():
    class DummyGraph:
        nodes = {"node1": "node1"}
        inputs = []

    class DummyData:
        id = "node1"
        concrete_value = None

    cfg = TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu"))

    class MyTensor(Tensor):
        @property
        def data(self):
            return DummyData()

    t_val_none = MyTensor(None, cfg)
    t_val_none._data = None

    class MockTensor(Tensor):
        @property
        def requires_grad(self):
            return False

        @property
        def trainable(self):
            return False

    t_skip = MockTensor(DummyData(), cfg)

    class DummyData2:
        id = "node2"

    t_skip2 = Tensor(DummyData2(), cfg)

    # 310->309: hasattr(t, "data") but NOT hasattr(t.data, "id")
    class DummyDataNoId:
        pass

    t_no_id = Tensor(DummyDataNoId(), cfg)

    tensors = [t_val_none, t_skip, t_skip2, t_no_id]

    _get_inputs_dict(DummyGraph())
    _find_wrt_tensors(DummyGraph())

    class DummyNode:
        shape = [2, 2]

    DummyGraph.nodes = {"node1": DummyNode()}
    _generate_dummy_input(DummyGraph(), "node1")

    t_orig = Tensor(np.array([1], dtype=np.int16), TensorConfig(shape=(1,), dtype=DType("int16"), device=Device("cpu")))
    _to_original_type(np.array([1.0], dtype=np.float32), t_orig)
    _to_original_type(np.array([1.0], dtype=bool), t_orig)

    _convert_to_tensors((np.array([1.0], dtype=np.float32),))
    _convert_to_tensors((np.array([True], dtype=np.bool_),))

    t_val = Tensor(np.array([1]), cfg)
    _get_concrete_val(t_val)
