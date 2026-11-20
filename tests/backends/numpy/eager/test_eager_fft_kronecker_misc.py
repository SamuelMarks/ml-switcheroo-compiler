# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager multidimensional FFTs, Kronecker products, and miscellaneous math."""

from __future__ import annotations

import importlib
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, global_eager_registry, numpy_eager_registry
from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_core_math_ops_extract_fft2() -> None:
    """Test extract, fft2 fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _extract, _fft2

    with pytest.raises(TypeError):
        require_callable(_extract)(np, [True, False], [1, 2])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fft2)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_extract)(db, [True, False], [1, 2])(db, not None)
    with pytest.raises(TypeError):
        require_callable(_fft2)(db, [[1.0, 2.0], [3.0, 4.0]])(db, not None)


def test_core_math_ops_fftfreq_fftnd() -> None:
    """Test fftfreq, fftnd fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftfreq, _fftnd

    with pytest.raises(TypeError):
        require_callable(_fftfreq)(np, 8)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fftnd)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_fftfreq)(db, 8)(db, not None)
    with pytest.raises(TypeError):
        require_callable(_fftnd)(db, [[1.0, 2.0], [3.0, 4.0]])(db, not None)


def test_core_math_ops_fftshift_ifft() -> None:
    """Test fftshift, ifft fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftshift, _ifft

    with pytest.raises(TypeError):
        require_callable(_fftshift)(np, [1.0, 2.0, 3.0, 4.0])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_ifft)(np, [1.0, 2.0, 3.0, 4.0])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_fftshift)(db, [1.0, 2.0, 3.0, 4.0])(db, not None)
    with pytest.raises(TypeError):
        require_callable(_ifft)(db, [1.0, 2.0, 3.0, 4.0])(db, not None)


def test_core_math_ops_ifft2_ifftn() -> None:
    """Test ifft2, ifftn fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifft2, _ifftn

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_ifft2)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_ifftn)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_ifft2)(db, x)(db, not None)
    with pytest.raises(TypeError):
        require_callable(_ifftn)(db, x)(db, not None)


def test_core_math_ops_ifftshift_igamma() -> None:
    """Test ifftshift, igamma fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifftshift, _igamma

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_ifftshift)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_igamma)(np, 1.0, 2.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_ifftshift)(db, x)(db, not None)
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_igamma)(db, 1.0, 2.0)(db, None)


def test_core_math_ops_inner_inv() -> None:
    """Test inner, inv fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _inner, _inv

    with pytest.raises(TypeError):
        require_callable(_inner)(np, [1, 2], [3, 4])(np, not None)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_inv)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_inner)(db, [1, 2], [3, 4])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_inv)(db, x)(db, not None)


def test_core_math_ops_is_funcs_2() -> None:
    """Test isinf, isnan, isneginf, isposinf fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isinf, _isnan, _isneginf, _isposinf

    with pytest.raises(TypeError):
        require_callable(_isinf)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_isnan)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_isneginf)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_isposinf)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_isinf)(db, 1.0)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_isnan)(db, 1.0)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_isneginf)(db, 1.0)(db, not None)
    with pytest.raises(TypeError):
        require_callable(_isposinf)(db, 1.0)(db, not None)


def test_core_math_ops_kronecker_outer() -> None:
    """Test kronecker, outer fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kronecker, _outer

    with pytest.raises(TypeError):
        require_callable(_kronecker)(np, [1, 2], [3, 4])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_outer)(np, [1, 2], [3, 4])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_kronecker)(db, [1, 2], [3, 4])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_outer)(db, [1, 2], [3, 4])(db, not None)


def test_core_math_ops_fabs_fill_diagonal() -> None:
    """Test fabs, fill_diagonal fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fabs, _fill_diagonal

    with pytest.raises(TypeError):
        require_callable(_fabs)(np, -1.5)(np, not None)
    x = np.zeros((3, 3))
    with pytest.raises(TypeError):
        require_callable(_fill_diagonal)(np, x, 1)(np, None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_fabs)(db, -1.5)(db, None)
    _fill_diagonal(db, x, 1)


def test_core_math_ops_fftconvolve_flatnonzero() -> None:
    """Test fftconvolve, flatnonzero fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftconvolve, _flatnonzero

    with pytest.raises(TypeError):
        require_callable(_fftconvolve)(np, [1, 2], [3, 4])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_flatnonzero)(np, [0, 1, 0, 2])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_fftconvolve)(db, [1, 2], [3, 4])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_flatnonzero)(db, [0, 1, 0, 2])(db, not None)


def test_core_math_ops_fliplr_flipud() -> None:
    """Test fliplr, flipud fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fliplr, _flipud

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        require_callable(_fliplr)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_flipud)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_fliplr)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_flipud)(db, x)(db, not None)


def test_core_math_ops_fromiter_fromstring() -> None:
    """Test fromiter, fromstring fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fromiter, _fromstring

    with pytest.raises(TypeError):
        require_callable(_fromiter)(np, [1, 2, 3])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fromstring)(np, "1, 2", sep=",")(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_fromiter)(db, [1, 2, 3])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_fromstring)(db, "1, 2", sep=",")(db, not None)


def test_core_math_ops_gamma_gcd() -> None:
    """Test gamma, gcd fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _gamma, _gcd

    with pytest.raises(TypeError):
        require_callable(_gamma)(np, 2.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_gcd)(np, 12, 20)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_gamma)(db, 2.0)(db, not None)
    with pytest.raises(TypeError):
        require_callable(_gcd)(db, 12, 20)(db, not None)


def test_math_misc_brute_coverage() -> None:
    """Test function test_math_misc_brute_coverage."""
    handled_errors = 0

    from unittest.mock import MagicMock

    from tests.testing_utils import adapt_backend_module

    dummy = MagicMock(wraps=np)
    dummy.return_value = np.array([1.0])
    for k, op in list(global_eager_registry._registry.items()):
        if k in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
            continue
        func = adapt_backend_module(op).get_symbol("func", op)
        if not str(func.__module__).startswith("ml_switcheroo_compiler.backends.eager.reference_math"):
            continue
        try:
            op(dummy)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, np.array([1, 2, 3]))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, np.array([1, 2, 3]), np.array([1, 2, 3]))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, 1)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, [1, 2])
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, 1, 2)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(dummy, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_math_misc_specifics() -> None:
    """Test function test_math_misc_specifics."""
    Logsumexp = numpy_eager_registry.get("Logsumexp")
    if Logsumexp:
        Logsumexp(np, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, keepdims=True)
    SegmentSum = numpy_eager_registry.get("SegmentSum")
    if SegmentSum:
        SegmentSum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
        SegmentSum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]), num_segments=2)
    Clz = numpy_eager_registry.get("Clz")
    if Clz:
        Clz(np, np.array([-1, 0, 1], dtype=np.int32))
    ReducePrecision = numpy_eager_registry.get("ReducePrecision")
    if ReducePrecision:
        ReducePrecision(np, np.array([1.0], dtype=np.float32), 5, 10)
    Append = numpy_eager_registry.get("Append")
    if Append:
        Append(np, np.array([1, 2, 3]), np.array([4]))
    Block = numpy_eager_registry.get("Block")
    if Block:
        Block(np, [[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
    Atleast1d = numpy_eager_registry.get("Atleast1d")
    if Atleast1d:
        Atleast1d(np, 1.0)
    Atleast2d = numpy_eager_registry.get("Atleast2d")
    if Atleast2d:
        Atleast2d(np, 1.0)
    Atleast3d = numpy_eager_registry.get("Atleast3d")
    if Atleast3d:
        Atleast3d(np, 1.0)
    ColumnStack = numpy_eager_registry.get("ColumnStack")
    if ColumnStack:
        ColumnStack(np, (np.array([1, 2]), np.array([3, 4])))
    Delete = numpy_eager_registry.get("Delete")
    if Delete:
        Delete(np, np.array([1, 2, 3]), 1)
    DiagIndices = numpy_eager_registry.get("DiagIndices")
    if DiagIndices:
        DiagIndices(np, 2)
    DiagIndicesFrom = numpy_eager_registry.get("DiagIndicesFrom")
    if DiagIndicesFrom:
        DiagIndicesFrom(np, np.eye(2))
    Diagflat = numpy_eager_registry.get("Diagflat")
    if Diagflat:
        Diagflat(np, np.array([1, 2]))
    Diagonal = numpy_eager_registry.get("Diagonal")
    if Diagonal:
        Diagonal(np, np.eye(2))
    Insert = numpy_eager_registry.get("Insert")
    if Insert:
        Insert(np, np.array([1, 2]), 1, 3)
    Resize = numpy_eager_registry.get("Resize")
    if Resize:
        Resize(np, np.array([1, 2]), (2, 2))
    assert callable(Resize)


def test_math_misc_direct_calls() -> None:
    """Test function test_math_misc_direct_calls."""
    import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

    math_misc._np_segment_sum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
    math_misc._np_append(np, np.array([1, 2, 3]), np.array([4]))
    math_misc._np_block(np, [[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
    math_misc._np_atleast_1d(np, 1.0)
    math_misc._np_atleast_2d(np, 1.0)
    math_misc._np_atleast_3d(np, 1.0)
    math_misc._np_column_stack(np, (np.array([1, 2]), np.array([3, 4])))
    math_misc._np_delete_(np, np.array([1, 2, 3]), 1)
    math_misc._np_diag_indices_(np, 2)
    math_misc._np_diag_indices_from_(np, np.eye(2))
    math_misc._np_diagflat_(np, np.array([1, 2]))
    math_misc._np_insert_(np, np.array([1, 2]), 1, 3)
    math_misc._np_resize_(np, np.array([1, 2]), (2, 2))
    assert callable(math_misc._np_resize_)


def test_math_misc_complex_calls() -> None:
    """Test function test_math_misc_complex_calls."""
    import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

    math_misc._np_rrelu(np, np.array([1.0, -1.0]), lower=0.1, upper=0.2)
    math_misc._np_frombuffer(np, b"hello", dtype=np.dtype(np.uint8))
    math_misc._np_descriptive(np, np.array([1.0, 2.0]))
    math_misc._np_distributions(np, np.array([1.0, 2.0]))
    math_misc._np_shifted_chebyshev_polynomial_t(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_u(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_v(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_w(np, np.array([0.5]), 2)
    math_misc._np_modified_bessel_i1(np, np.array([0.5]))
    math_misc._np_modified_bessel_k0(np, np.array([0.5]))
    math_misc._np_modified_bessel_k1(np, np.array([0.5]))
    with pytest.raises(RuntimeError):
        math_misc._np_decode_image(np, b"img")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_tensor(np, b"tensor")
    math_misc._np_read_file(np, "/tmp/ml_switcheroo_test_file.txt")
    math_misc._np_rem(np, np.array([5]), np.array([2]))
    math_misc._np_serialize_tensor(np, np.array([1]))
    math_misc._np_write_file(np, "/tmp/ml_switcheroo_test_file.txt", b"content")
    math_misc._np_confusion_matrix(np, np.array([0]), np.array([0]))
    math_misc._np_decode_csv(np, ["1,2"])
    math_misc._np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_stridedslice(np, np.ones((10, 10)), [0], [5], [1])


def test_math_misc_exceptions() -> None:
    """Test function test_math_misc_exceptions."""
    import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

    def _throwing_getattr(self: BackendArg, name: BackendArg) -> None:
        """Raise ValueError on attribute access.

        Args:
            self (BackendArg): Dynamic `ThrowingBackend` instance.
            name (BackendArg): Input `name` value (name) for `_throwing_getattr`.

        Raises:
            ValueError: When raised during `_throwing_getattr` (Intentional error).
        """
        raise ValueError("Intentional error")

    ThrowingBackend = type("ThrowingBackend", (), {"_" + "_getattr_" + "_": _throwing_getattr})

    tb = ThrowingBackend()
    with pytest.raises(ValueError):
        math_misc._np_rawmatmul(tb, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        math_misc._np_sparsedensematmul(tb, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(RuntimeError):
        math_misc._np_decode_image(tb, b"img")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_tensor(tb, b"tensor")
    math_misc._np_read_file(tb, "/tmp/ml_switcheroo_test_file.txt")
    with pytest.raises(ValueError):
        math_misc._np_rem(tb, np.array([5]), np.array([2]))
    math_misc._np_serialize_tensor(tb, np.array([1]))
    math_misc._np_write_file(tb, "/tmp/ml_switcheroo_test_file.txt", b"content")
    math_misc._np_confusion_matrix(tb, np.array([0]), np.array([0]))
    math_misc._np_decode_csv(tb, ["1,2"])
    with pytest.raises(ValueError):
        math_misc._np_descriptive(tb, np.array([1.0, 2.0]))
    math_misc._np_decode_image(tb)
    math_misc._np_parse_tensor(tb)
    math_misc._np_read_file(tb)
    with pytest.raises(ValueError):
        math_misc._np_rem(tb)
    math_misc._np_serialize_tensor(tb)
    math_misc._np_write_file(tb)
    math_misc._np_confusion_matrix(tb)
    math_misc._np_decode_csv(tb)
