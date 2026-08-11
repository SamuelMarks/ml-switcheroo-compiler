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


def test_group_norm_positive_axis_coverage():
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_group_ops import _group_norm

    x = np.random.randn(2, 4, 4, 8).astype(np.float32)
    out = _group_norm(np, x, 2, axis=3, epsilon=1e-05)
    reshaped = x.reshape(2, 4, 4, 2, 4)
    mean = np.mean(reshaped, axis=(1, 2, 4), keepdims=True)
    var = np.var(reshaped, axis=(1, 2, 4), keepdims=True)
    expected = ((reshaped - mean) / np.sqrt(var + 1e-05)).reshape(x.shape)
    np.testing.assert_allclose(out, expected, rtol=1e-05, atol=1e-05)
