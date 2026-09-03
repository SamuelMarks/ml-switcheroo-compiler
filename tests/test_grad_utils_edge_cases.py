def test_grad_utils_extra():
    from unittest.mock import patch

    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.errors import SwitcherooError
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.utils import _check_scalar, _get_concrete_val, _get_inputs_dict, _to_original_type

    class MockTensor:
        shape = (1, "a")

    with pytest.raises(SwitcherooError, match="backward"):
        _check_scalar(MockTensor())

    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    class MockTensorProxy:
        _data = ProxyTensor(None, (1,))

    assert _get_concrete_val(MockTensorProxy()) is None

    class MockProxy(ProxyTensor):
        def __init__(self):
            pass

    mock_proxy = MockProxy()
    mock_proxy.concrete_value = 5.0
    mock_tensor_proxy2 = MockTensorProxy()
    mock_tensor_proxy2._data = mock_proxy
    assert _get_concrete_val(mock_tensor_proxy2) == 5.0

    class MockGraph:
        inputs = ["in1"]
        nodes = {}

    with pytest.raises(ValueError, match="Missing input"):
        _get_inputs_dict(MockGraph())

    t = Tensor(1.0, TensorConfig((1,), "float32", "cpu"))
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.asarray.return_value = np.array([1.0], dtype=np.float64)
        res = _to_original_type(1.0, t)
        assert getattr(res, "dtype", None).name == "Float64"

        mock_backend.return_value.asarray.return_value = np.array([1], dtype=np.int32)
        res = _to_original_type(1, t)
        assert getattr(res, "dtype", None).name == "Int32"

        mock_backend.return_value.asarray.return_value = np.array([True], dtype=bool)
        res = _to_original_type(True, t)
        assert getattr(res, "dtype", None).name == "Bool"
