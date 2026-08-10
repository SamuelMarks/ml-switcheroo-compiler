def test_invoke_grouped_op_coverage():
    from ml_switcheroo_compiler.backends.eager.core_group_ops import _invoke_grouped_op

    class TorchMock:
        def mean(self, x, **kwargs):
            if "axis" in kwargs:
                raise TypeError()
            return "torch_mean"

        def var(self, x, **kwargs):
            if "axis" in kwargs:
                raise TypeError()
            return "torch_var"

    assert _invoke_grouped_op(TorchMock(), "mean", None, None) == "torch_mean"
    assert _invoke_grouped_op(TorchMock(), "variance", None, None) == "torch_var"

    import pytest

    with pytest.raises(ValueError):
        _invoke_grouped_op(TorchMock(), "unknown", None, None)
