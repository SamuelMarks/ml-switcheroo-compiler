"""Behavioral unit tests for exponential."""

from __future__ import annotations

from unittest.mock import patch

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.unary.base import UnaryMathOp
from ml_switcheroo_compiler.ops.unary.exponential import Exp, Exp2, Expm1, Log, Log1P, Log10, NanToNum, ZeroFraction


def test_nan_to_num_call(mocker: MockerFixture) -> None:
    """Test function test_nan_to_num_call.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    op = NanToNum()
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    import sys

    sys.modules["ml_switcheroo_compiler.ops.base"] = sys.modules["ml_switcheroo_compiler.ops.base"]
    mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")
    mocker.patch("ml_switcheroo_compiler.ops.base.dispatch_op", return_value="result", create=True)
    out = op(x=Tensor([1], TensorConfig((1,), "float32", "cpu")), copy=True)
    assert out == "result"


def test_unary_exponential_ops() -> None:
    """Verify all exponential and logarithmic unary operation definitions."""
    ops: list[tuple[UnaryMathOp, str]] = [
        (Exp(), "Exp"),
        (Log(), "Log"),
        (Exp2(), "Exp2"),
        (Expm1(), "Expm1"),
        (Log10(), "Log10"),
        (Log1P(), "Log1P"),
        (NanToNum(), "NanToNum"),
        (ZeroFraction(), "ZeroFraction"),
    ]
    for inst, name in ops:
        assert inst.op_name == name

    assert Exp2().np_op_name == "exp2"
    assert Expm1().np_op_name == "expm1"
    assert Log10().np_op_name == "log10"
    assert Log1P().np_op_name == "log1p"

    # Test NanToNum copy argument handling
    with patch("ml_switcheroo_compiler.ops.base.dispatch_op", return_value="mock_nanto_num") as mock_dispatch:
        res = NanToNum()(10, copy=True)
        assert res == "mock_nanto_num"
        mock_dispatch.assert_called_once_with("NanToNum", 10)

        mock_dispatch.reset_mock()
        res_no_copy = NanToNum()(10)
        assert res_no_copy == "mock_nanto_num"
        mock_dispatch.assert_called_once_with("NanToNum", 10)
