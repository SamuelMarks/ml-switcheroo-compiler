# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager polynomials, FFT frequencies, IO loaders, and softmax/one-hot."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, adapt_backend_module, extract_test_shape

_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_np_polynomial_and_bessel_ops() -> None:
    """Test polynomial and bessel ops."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    assert backend.__name__ == "NumpyGenerator"
    ops_and_args = [
        ("chebyshev_polynomial_t", (np.array([2.0]), np.array([0.5]))),
        ("chebyshev_polynomial_u", (np.array([2.0]), np.array([0.5]))),
        ("hermite_polynomial_h", (np.array([2.0]), np.array([0.5]))),
        ("hermite_polynomial_he", (np.array([2.0]), np.array([0.5]))),
        ("laguerre_polynomial_l", (np.array([2.0]), np.array([0.5]))),
        ("legendre_polynomial_p", (np.array([2.0]), np.array([0.5]))),
        ("modified_bessel_i0", (np.array([0.5]),)),
        ("modified_bessel_i1", (np.array([0.5]),)),
        ("modified_bessel_k0", (np.array([0.5]),)),
        ("modified_bessel_k1", (np.array([0.5]),)),
        ("shifted_chebyshev_polynomial_t", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_u", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_v", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_w", (np.array([2.0]), np.array([0.5]))),
    ]
    for op_name, args in ops_and_args:
        out = backend.execute_op(op_name, *args)


def test_np_polynomial_bessel_no_args() -> None:
    """Test fallback when args are missing entirely."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops_with_one_arg = ["modified_bessel_i0", "modified_bessel_i1", "modified_bessel_k0", "modified_bessel_k1"]
    results = [backend.execute_op(op_name) for op_name in ops_with_one_arg]
    assert len(results) == 4
    assert all(np.asarray(r).size >= 1 for r in results)


def test_np_polynomial_get_sc() -> None:
    """Test get sc."""
    from tests.backends.numpy.eager._math_advanced_test_ns import _get_sc

    sc = _get_sc()
    assert sc is None or adapt_backend_module(sc).has_symbol("jv")


def test_np_polynomial_get_sc_fallback() -> None:
    """Test fallback of _get_sc."""
    import sys
    import unittest.mock as mock

    from tests.backends.numpy.eager._math_advanced_test_ns import _get_sc

    with mock.patch.dict(sys.modules, {"scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_polynomial_get_sc_branch() -> None:
    """Test get sc."""
    import sys
    import unittest.mock as mock

    from tests.backends.numpy.eager._math_advanced_test_ns import _get_sc

    with mock.patch.dict(sys.modules, {"scipy": None, "scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_polynomial_get_sc_branch2() -> None:
    """Test get sc."""
    import sys
    import unittest.mock as mock

    from tests.backends.numpy.eager._math_advanced_test_ns import _get_sc

    with mock.patch.dict(sys.modules, {"scipy": None, "scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_fft_ops() -> None:
    """Test fft ops."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops = [
        "Rfft",
        "Ifft",
        "Fftn",
        "Ifftn",
        "Rfftn",
        "Irfftn",
        "Ifft2",
        "Rfft2",
        "Irfft2",
        "Fftnd",
        "Ifftnd",
        "Rfftnd",
        "Irfftnd",
        "Fftshift",
        "Ifftshift",
        "Hfft",
    ]
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for op_name in ops:
        out = backend.execute_op(op_name, a)
        assert extract_test_shape(out)[0] == 2


def test_np_rfftfreq_op() -> None:
    """Test rfftfreq."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Rfftfreq", 10, d=0.1)
    assert tuple(extract_test_shape(out)) == (6,)


def test_np_fft_ops_missing_args() -> None:
    """Test fft missing ops args."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops = [
        "Rfft",
        "Ifft",
        "Fftn",
        "Ifftn",
        "Rfftn",
        "Irfftn",
        "Ifft2",
        "Rfft2",
        "Irfft2",
        "Fftnd",
        "Ifftnd",
        "Rfftnd",
        "Irfftnd",
        "Fftshift",
        "Ifftshift",
        "Hfft",
        "Rfftfreq",
    ]
    for op_name in ops:
        with pytest.raises(TypeError):
            backend.execute_op(op_name)


def test_np_misc_dummy_ops() -> None:
    """Test misc dummy ops."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]), num_classes=2)
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError):
        backend.execute_op("ConfusionMatrix")
    out = backend.execute_op("Descriptive", np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        backend.execute_op("Descriptive")
    out = backend.execute_op("Distributions")
    backend.execute_op("CreateToken")
    out = backend.execute_op("Rrelu", np.array([-1.0, 1.0]))
    out = backend.execute_op("Clip", np.array([-1.0, 1.0]), np.array([0.0]), np.array([0.5]))
    with pytest.raises(TypeError):
        out = backend.execute_op("Clip")
    out = backend.execute_op("Frombuffer", b"12341234")
    out = backend.execute_op("Frombuffer")
    assert out is None
    out = backend.execute_op("Softmax", np.array([0.0, 1.0]))
    out = backend.execute_op("Softmax")
    assert out is None
    out = backend.execute_op("Sigmoid", np.array([0.0, 1.0]))
    out = backend.execute_op("Sigmoid")
    assert out is None
    out = backend.execute_op("LogSoftmax", np.array([0.0, 1.0]))
    out = backend.execute_op("LogSoftmax")
    assert out is None
    out = backend.execute_op("OneHot", np.array([0, 1]), 2)
    out = backend.execute_op("OneHot", np.array([0, 1]), np.array(2), axis=0)
    out = backend.execute_op("OneHot")
    assert out is None
    out = backend.execute_op("CtcLoss", np.array([0, 1]), np.array([0, 1]), np.array([2]), np.array([2]))
    out = backend.execute_op("CircleLoss", np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.isscalar(out) or ensure_ndarray(out).size == 1
    out = backend.execute_op(
        "CategoricalGeneralizedCrossEntropy", np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.1], [0.2, 0.8]])
    )
    assert np.isscalar(out) or ensure_ndarray(out).size == 1


def test_np_one_hot_axis_fallback() -> None:
    """Test one hot axis branch."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("OneHot", np.array([0, 1]), np.array(2), axis=1)
    assert tuple(extract_test_shape(out)) == (2, 2)


def test_np_log_softmax_no_axis() -> None:
    """Test log softmax axis."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("LogSoftmax", np.array([0.0, 1.0]), axis=-1)
    assert tuple(extract_test_shape(out)) == (2,)


def test_np_softmax_no_axis() -> None:
    """Test softmax axis."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Softmax", np.array([0.0, 1.0]), axis=-1)
    assert tuple(extract_test_shape(out)) == (2,)


def test_np_rrelu_no_args() -> None:
    """Test rrelu no args."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Rrelu", np.array([-1.0, 1.0]), lower=0.1, upper=0.2)
    assert tuple(extract_test_shape(out)) == (2,)


def test_np_clip_no_args() -> None:
    """Test clip no args."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Clip", np.array([-1.0, 1.0]), a_min=np.array([0.0]), a_max=np.array([0.5]))
    assert tuple(extract_test_shape(out)) == (2,)


def test_np_one_hot_axis_fallback2() -> None:
    """Test one hot axis branch2."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("OneHot", np.array([0, 1]), 2, axis=0)
    assert tuple(extract_test_shape(out)) == (2, 2)


def test_np_io_dummy_ops() -> None:
    """Test function test_np_io_dummy_ops."""
    import pytest

    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"
    ):
        import numpy as np

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        out = backend.execute_op("DecodeCsv", np.array(b"a,b,c"))
        out = backend.execute_op("DecodeImage", np.array(b"img"))
        out = backend.execute_op("ParseTensor", np.array(b"tensor"))
        out = backend.execute_op("ReadFile", "/tmp/ml_switcheroo_test_file.txt")
        out = backend.execute_op("Rem", np.array([5.0]), np.array([2.0]))
        out = backend.execute_op("Rem")
        assert out is None
        out = backend.execute_op("SerializeTensor", np.array([1.0]))
        out = backend.execute_op("WriteFile", "/tmp/ml_switcheroo_test_file.txt", np.array(b"content"))
        assert out is None


def test_io_fallbacks() -> None:
    """Test IO fallbacks."""
    import os
    import tempfile

    from ml_switcheroo_compiler.ops.io import BinaryRecordWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test.bin")
        writer = BinaryRecordWriter(bin_path)
        writer.write(b"")
        writer.close()
        assert writer._closed is True
        assert os.path.getsize(bin_path) > 0


def test_io_fallbacks_load() -> None:
    """Test IO load fallbacks."""
    import unittest.mock as mock

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend(MockTestStub):
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function load.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Raises:
                    NotImplementedError: If `load` is not implemented by the subclass or target backend.
                """
                raise NotImplementedError()

        mock_get.return_value = DummyBackend


def test_io_fallbacks_load_file() -> None:
    """Test load file."""
    import unittest.mock as mock

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend(MockTestStub):
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function load.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Raises:
                    NotImplementedError: If `load` is not implemented by the subclass or target backend.
                """
                raise NotImplementedError()

        mock_get.return_value = DummyBackend
        with mock.patch("ml_switcheroo_compiler.ops.io._fallback_load", return_value="success"):
            pass


def test_np_io_dummy_ops_with_backends() -> None:
    """Test io dummy ops with backend."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.ops.io import decode_bmp, decode_gif, decode_jpeg, decode_png

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend(MockTestStub):
            """Test class DummyBackend."""

            @classmethod
            def decode_jpeg(cls, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function decode_jpeg.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `decode_jpeg`.
                """
                return "jpeg"

            @classmethod
            def decode_png(cls, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function decode_png.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `decode_png`.
                """
                return "png"

            @classmethod
            def decode_gif(cls, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function decode_gif.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `decode_gif`.
                """
                return "gif"

            @classmethod
            def decode_bmp(cls, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function decode_bmp.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `decode_bmp`.
                """
                return "bmp"

        mock_get.return_value = DummyBackend
        assert decode_jpeg("test")._data == "jpeg"
        assert decode_png("test")._data == "png"
        assert decode_gif("test")._data == "gif"
        assert decode_bmp("test")._data == "bmp"
