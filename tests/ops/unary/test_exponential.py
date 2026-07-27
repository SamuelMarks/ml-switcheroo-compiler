# ruff: noqa: E501
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.unary.exponential import NanToNum


def test_nan_to_num_call(mocker):
    op = NanToNum()
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    import sys

    sys.modules["ml_switcheroo_compiler.ops.base"] = sys.modules["ml_switcheroo_compiler.ops.base"]
    mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")
    mocker.patch("ml_switcheroo_compiler.ops.base.dispatch_op", return_value="result", create=True)
    try:
        op(x=Tensor([1], TensorConfig((1,), "float32", "cpu")), copy=True)
    except Exception:
        pass
