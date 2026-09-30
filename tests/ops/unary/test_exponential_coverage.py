"""Tests for test_exponential_coverage."""

from __future__ import annotations

from unittest.mock import patch

from ml_switcheroo_compiler.ops.unary.exponential import Exp, Exp2, Expm1, Log, Log1P, Log10, NanToNum, ZeroFraction


def test_unary_exponential_ops() -> None:
    """Verify all exponential and logarithmic unary operation definitions."""
    ops = [
        (Exp(), "Exp", None),
        (Log(), "Log", None),
        (Exp2(), "Exp2", "exp2"),
        (Expm1(), "Expm1", "expm1"),
        (Log10(), "Log10", "log10"),
        (Log1P(), "Log1P", "log1p"),
        (NanToNum(), "NanToNum", None),
        (ZeroFraction(), "ZeroFraction", None),
    ]
    for inst, name, np_name in ops:
        assert inst.op_name == name
        if np_name:
            assert getattr(inst, "np_op_name", None) == np_name

    # Test NanToNum copy argument handling
    with patch("ml_switcheroo_compiler.ops.base.dispatch_op", return_value="mock_nanto_num") as mock_dispatch:
        res = NanToNum()(10, copy=True)
        assert res == "mock_nanto_num"
        mock_dispatch.assert_called_once_with("NanToNum", 10)

        mock_dispatch.reset_mock()
        res_no_copy = NanToNum()(10)
        assert res_no_copy == "mock_nanto_num"
        mock_dispatch.assert_called_once_with("NanToNum", 10)
