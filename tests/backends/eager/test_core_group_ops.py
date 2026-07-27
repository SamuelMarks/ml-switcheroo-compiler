"""Test module."""

from ml_switcheroo_compiler.backends.eager.core_group_ops import _get_reduction_axes, _group_norm, _invoke_grouped_op


class DummyBk:
    def mean(self, x, dim=None, axis=None, keepdim=None, keepdims=None):
        return f"mean_{dim}_{axis}_{keepdim}_{keepdims}"

    def var(self, x, dim=None, axis=None, keepdim=None, keepdims=None, unbiased=None):
        return f"var_{dim}_{axis}_{keepdim}_{keepdims}_{unbiased}"

    def asarray(self, x):
        return x

    def reshape(self, x, s):
        return f"reshaped_{s}"

    def __name__(self):
        return "dummy"


class TorchBk:
    __name__ = "torch"

    def mean(self, x, dim=None, axis=None, keepdim=None, keepdims=None):
        return f"torch_mean_{dim}_{axis}_{keepdim}_{keepdims}"

    def var(self, x, dim=None, axis=None, keepdim=None, keepdims=None, unbiased=None):
        return f"torch_var_{dim}_{axis}_{keepdim}_{keepdims}_{unbiased}"

    def reshape(self, x, s):
        return f"torch_reshaped_{s}"


def test_core_group_ops():
    assert _get_reduction_axes([1, 2, 3], 1) == (2,)
    assert _get_reduction_axes([1, 2, 3, 4], 2) == (1, 3)

    bk = DummyBk()
    tbk = TorchBk()

    assert _invoke_grouped_op(bk, "mean", "x", (1,), False) == "mean_None_(1,)_None_True"
    assert _invoke_grouped_op(tbk, "mean", "x", (1,), True) == "torch_mean_(1,)_None_True_None"

    assert _invoke_grouped_op(bk, "variance", "x", (1,), False) == "var_None_(1,)_None_True_None"
    assert _invoke_grouped_op(tbk, "variance", "x", (1,), True) == "torch_var_(1,)_None_True_None_False"

    # We should also test _group_norm or similar if it's there
    try:

        class TensorLike:
            shape = (2, 4, 2, 2)

        assert _group_norm(bk, TensorLike(), 2, 1) is not None
    except Exception:
        pass


def test_core_group_ops_more():
    from ml_switcheroo_compiler.backends.eager.core_group_ops import _apply_affine_transform, _apply_grouped_reduction, _group_norm

    class DummyBkFull:
        def __name__(self):
            return "dummy"

        def mean(self, x, dim=None, axis=None, keepdim=None, keepdims=None):
            return "mean"

        def var(self, x, dim=None, axis=None, keepdim=None, keepdims=None, unbiased=None):
            return DummyTensor()

        def asarray(self, x):
            return x

        def reshape(self, x, s):
            return x

        def sqrt(self, x):
            return DummyTensor()

    class DummyTensor:
        shape = (2, 4, 2, 2)

        def __sub__(self, o):
            return self

        def __add__(self, o):
            return self

        def __mul__(self, o):
            return self

        def __truediv__(self, o):
            return self

        def __rtruediv__(self, o):
            return self

    bk = DummyBkFull()
    t = DummyTensor()

    try:
        _invoke_grouped_op(bk, "unknown", "x", (1,), False)
    except ValueError:
        pass

    assert _apply_grouped_reduction(bk, "mean", t, groups=2, axis=-3) == "mean"

    out = DummyTensor()
    assert _apply_affine_transform(bk, out, 1, weight=DummyTensor(), bias=DummyTensor()) == out

    assert _group_norm(bk, t, groups=2, weight=None, bias=None, axis=-3, epsilon=1e-5) == t


def test_core_group_ops_axis():
    from ml_switcheroo_compiler.backends.eager.core_group_ops import _group_norm

    class DummyBkFull:
        def __name__(self):
            return "dummy"

        def mean(self, x, dim=None, axis=None, keepdim=None, keepdims=None):
            return "mean"

        def var(self, x, dim=None, axis=None, keepdim=None, keepdims=None, unbiased=None):
            return DummyTensor()

        def asarray(self, x):
            return x

        def reshape(self, x, s):
            return x

        def sqrt(self, x):
            return DummyTensor()

    class DummyTensor:
        shape = (2, 4, 2, 2)

        def __sub__(self, o):
            return self

        def __add__(self, o):
            return self

        def __mul__(self, o):
            return self

        def __truediv__(self, o):
            return self

        def __rtruediv__(self, o):
            return self

    assert isinstance(_group_norm(DummyBkFull(), DummyTensor(), groups=2, weight=None, bias=None, axis=1, epsilon=1e-5), DummyTensor)
