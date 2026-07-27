"""Test module."""

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _from_numpy_array, _jax_from_numpy, _mlx_from_numpy, _to_channels_last, _to_numpy_array, _torch_from_numpy


class DummyNp:
    def asarray(self, x):
        return "np_asarray_res"

    def array(self, out, dtype=None):
        if dtype:
            return f"np_array_res_{dtype}"
        return "np_array_res"


class DummyTensorWithNumpy:
    def numpy(self):
        return "numpy_res"


class DummyTensorTorch:
    def detach(self):
        class Cpu:
            def cpu(self):
                class Numpy:
                    def numpy(self):
                        return "torch_numpy_res"

                return Numpy()

        return Cpu()


class DummyOriginal:
    dtype = "float32"


def test_to_numpy_array():
    np_mod = DummyNp()

    t1 = DummyTensorWithNumpy()
    assert _to_numpy_array(np_mod, t1, "test") == "numpy_res"

    t2 = DummyTensorTorch()
    assert _to_numpy_array(np_mod, t2, "torch") == "torch_numpy_res"

    assert _to_numpy_array(np_mod, "other", "test") == "np_asarray_res"


def test_from_numpy_array():
    backend = DummyNp()

    assert _from_numpy_array(backend, "out", "torch") == "out"
    assert _from_numpy_array(backend, "out", "mlx.core") == "out"
    assert _from_numpy_array(backend, "out", "jax.numpy") == "out"

    assert _from_numpy_array(backend, "out", "other") == "np_array_res"

    orig = DummyOriginal()
    assert _from_numpy_array(backend, "out", "other", orig) == "np_array_res_float32"


def test_other_utils():
    assert _torch_from_numpy("out") == "out"
    assert _mlx_from_numpy("out") == "out"
    assert _jax_from_numpy("out") == "out"

    assert _to_channels_last(None, "imgs", None) == "imgs"
    assert _from_channels_last(None, "out", None) == "out"
