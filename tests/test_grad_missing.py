import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import SwitcherooError
from ml_switcheroo_compiler.core.tensor_mixins import ProxyTensor
from ml_switcheroo_compiler.grad import _check_scalar, _find_wrt_tensors, _generate_fallback_input, _get_concrete_val, _get_inputs_dict, _to_original_type


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape


def test_validate_scalar():
    with pytest.raises(SwitcherooError, match="backward.. can only be called on scalar tensors."):
        _check_scalar(DummyTensor(("a",)))


def test_get_concrete_val():
    class TestTensor:
        def __init__(self, data=None, _data=None):
            self.data = data
            self._data = _data

    # _data is ProxyTensor with concrete_value
    pt = ProxyTensor("id", (), "float32")
    pt.concrete_value = 42
    t1 = TestTensor(data=type("D", (), {"concrete_value": None})(), _data=pt)
    assert _get_concrete_val(t1) == 42

    # _data is normal value, not a ProxyTensor
    t2 = TestTensor(data=type("D", (), {"concrete_value": None})(), _data=100)
    assert _get_concrete_val(t2) == 100


def test_generate_fallback_input():
    from unittest.mock import patch

    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Input", shape_metadata=("a", 2), attributes={"dtype": "float64"})
    g.nodes["n1"] = n1

    with patch("ml_switcheroo_compiler.grad.get_active_backend") as mock_backend_getter:
        mock_backend = mock_backend_getter.return_value
        mock_backend.execute_op.return_value = "dummy_ones"

        res = _generate_fallback_input(g, "n1")
        assert res == "dummy_ones"
        mock_backend.execute_op.assert_called_with("Ones", (1, 2), dtype="float64")


def test_generate_fallback_input_no_node():
    from unittest.mock import patch

    g = LogicalGraph()
    with patch("ml_switcheroo_compiler.grad.get_active_backend") as mock_backend_getter:
        mock_backend = mock_backend_getter.return_value
        mock_backend.execute_op.return_value = "dummy_ones"

        res = _generate_fallback_input(g, "missing")
        assert res == "dummy_ones"
        mock_backend.execute_op.assert_called_with("Ones", (), dtype="float32")


def test_find_wrt_tensors():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable

    # We need to create a graph with nodes that match some tensors
    g = LogicalGraph()
    g.nodes["t1"] = LogicalNode(id="t1", op_type="Variable")
    g.nodes["t2"] = LogicalNode(id="t2", op_type="Constant")

    class FakeData:
        def __init__(self, id):
            self.id = id

    t_var = Variable(data=FakeData("t1"), config=TensorConfig((), "float32", None))
    t_const = Tensor(data=FakeData("t2"), config=TensorConfig((), "float32", None))
    # mock the property
    with patch("ml_switcheroo_compiler.core.tensor_mixins.TensorPropertiesMixin.requires_grad", new_callable=pytest.MonkeyPatch):
        t_const._requires_grad = True  # bypass property

    t_not_in_graph = Tensor(data=FakeData("t3"), config=TensorConfig((), "float32", None))
    t_not_in_graph._requires_grad = True

    t_no_req_grad = Tensor(data=FakeData("t2"), config=TensorConfig((), "float32", None))
    t_no_req_grad._requires_grad = False
    t_no_req_grad.trainable = False

    objs = [t_var, t_const, t_not_in_graph, t_no_req_grad]

    with patch("gc.get_objects", return_value=objs):
        wrt_tensors, wrt_ids = _find_wrt_tensors(g)
        assert len(wrt_tensors) == 2
        assert len(wrt_ids) == 2
        assert "t1" in wrt_ids
        assert "t2" in wrt_ids


def test_get_inputs_dict():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class MockGraph:
        def __init__(self):
            self.nodes = {"in1": LogicalNode(id="in1", op_type="Input")}
            self.inputs = ["in1", "in2"]

    g = MockGraph()

    class FakeData:
        def __init__(self, id):
            self.id = id

    t1 = Tensor(data=FakeData("in1"), config=TensorConfig((), "float32", None))
    # Add tensors to cover the missing branches in _get_inputs_dict (306->305, 308->305, 310->305)
    t_no_data = Tensor(data=None, config=TensorConfig((), "float32", None))
    t_no_id = Tensor(data=type("D", (), {})(), config=TensorConfig((), "float32", None))
    t_not_in_graph = Tensor(data=FakeData("not_in_graph"), config=TensorConfig((), "float32", None))
    t_no_val = Tensor(data=FakeData("in1"), config=TensorConfig((), "float32", None))  # we will mock _get_concrete_val to return None for this specific one

    with patch("gc.get_objects", return_value=[t1, t_no_data, t_no_id, t_not_in_graph, t_no_val]):
        with patch("ml_switcheroo_compiler.grad.get_active_backend") as mock_backend_getter:
            mock_backend = mock_backend_getter.return_value

            def mock_asarray(val):
                return val

            mock_backend.asarray.side_effect = mock_asarray
            mock_backend.execute_op.return_value = 1  # dummy 1

            def mock_get_concrete_val(t):
                if t is t_no_val:
                    return None
                return 42

            with patch("ml_switcheroo_compiler.grad._get_concrete_val", side_effect=mock_get_concrete_val):
                res = _get_inputs_dict(g)
                assert "in1" in res
                assert res["in1"] == 42
                assert "in2" in res
                assert res["in2"] == 1


def test_validate_and_convert_primals():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.grad import _convert_to_tensors

    with patch("ml_switcheroo_compiler.grad.get_active_backend") as mock_backend_getter:
        mock_backend = mock_backend_getter.return_value
        mock_backend.asarray.return_value = np.array(1.0, dtype=np.float32)

        primals = [1.0]
        res = _convert_to_tensors(primals)
        assert len(res) == 1
        assert res[0].dtype.value == "float32"


def test_to_original_type():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t_orig = Tensor(data=np.array(1), config=TensorConfig((1,), "float32", None))

    with patch("ml_switcheroo_compiler.grad.get_active_backend") as mock_backend_getter:
        mock_backend = mock_backend_getter.return_value

        # Test float64
        mock_backend.asarray.return_value = np.array(1.0, dtype=np.float64)
        res = _to_original_type(1.0, t_orig)
        assert res.dtype.value == "float64"

        # Test int32
        mock_backend.asarray.return_value = np.array(1, dtype=np.int32)
        res = _to_original_type(1, t_orig)
        assert res.dtype.value == "int32"

        # Test bool
        mock_backend.asarray.return_value = np.array(True, dtype=bool)
        res = _to_original_type(True, t_orig)
        assert res.dtype.value == "bool"

        # Test float32 (default)
        mock_backend.asarray.return_value = np.array(1.0, dtype=np.float32)
        res = _to_original_type(1.0, t_orig)
        assert res.dtype.value == "float32"

        # Test original type int
        res = _to_original_type(1.0, 5)
        assert res == 1

        # test original float
        res = _to_original_type(1.0, 5.0)
        assert isinstance(res, float)

        # test exception in typecast fallback
        res = _to_original_type(1.0, "not a number")
        assert res == 1.0


def test_backward_graph_missing_output():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyData:
        id = "dummy_id"

    # To pass _check_scalar, shape must be ()
    t_out = Tensor(data=DummyData(), config=TensorConfig((), "float32", None))

    class DummyGraph:
        def __init__(self):
            self.outputs = ["dummy_out"]
            self.nodes = {"dummy_id": type("N", (), {"id": "dummy_id", "op_type": "Constant"})}
            self.inputs = []

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state") as mock_state:
        mock_state.is_tracing = True
        # Properly mock the property active_graph
        mock_state.active_graph = DummyGraph()

        # Return two elements so we cover 376->374 taking a loop iteration
        with patch("ml_switcheroo_compiler.grad._find_wrt_tensors", return_value=([MagicMock(), MagicMock()], ["wrt_id_1", "wrt_id_2"])):
            with patch("ml_switcheroo_compiler.grad._get_inputs_dict", return_value={}):
                with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_build:
                    mock_grad_graph = MagicMock()
                    mock_grad_graph.outputs = ["grad_id_1", "grad_id_2"]
                    mock_build.return_value = mock_grad_graph

                    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph") as mock_eval:
                        # Only provide the second one
                        mock_eval.return_value = {"grad_id_2": 42.0}
                        from ml_switcheroo_compiler.grad import backward

                        backward(t_out)
