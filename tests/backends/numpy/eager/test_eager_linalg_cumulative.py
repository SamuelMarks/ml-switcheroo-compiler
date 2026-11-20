# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager matrix decompositions, linear solvers, and cumulative ops."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_float, coerce_numeric, require_callable

_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_core_math_ops_adjoint_det() -> None:
    """Test adjoint, det fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _adjoint, _det

    x = np.array([[1.0 + 1j, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_adjoint)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_det)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_adjoint)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_det)(db, x)(db, not None)

    class DummyBackend2(MockTestStub):
        """Test class DummyBackend2."""

        def conj(self, x: BackendArg) -> BackendArg:
            """Test function conj.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `conj`.
            """
            return np.conj(ensure_ndarray(x))

        def transpose(self, x: BackendArg) -> BackendArg:
            """Test function transpose.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `transpose`.
            """
            return np.transpose(ensure_ndarray(x))

    assert _adjoint(DummyBackend2(), x) is not None


def test_core_math_ops_eig_funcs() -> None:
    """Test eig, eigh, eigvals, eigvalsh fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _eig, _eigh, _eigvals, _eigvalsh

    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        require_callable(_eig)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_eigh)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_eigvals)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_eigvalsh)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_eig)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_eigh)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_eigvals)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_eigvalsh)(db, x)(db, not None)

    class DummyBackendLinalg(MockTestStub):
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def eig(x: BackendArg) -> BackendArg:
                """Test function eig.

                Args:
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `eig`.
                """
                return x

            @staticmethod
            def eigh(x: BackendArg) -> BackendArg:
                """Test function eigh.

                Args:
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `eigh`.
                """
                return x

            @staticmethod
            def eigvals(x: BackendArg) -> BackendArg:
                """Test function eigvals.

                Args:
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `eigvals`.
                """
                return x

            @staticmethod
            def eigvalsh(x: BackendArg) -> BackendArg:
                """Test function eigvalsh.

                Args:
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `eigvalsh`.
                """
                return x

    dbl = DummyBackendLinalg()
    assert _eig(dbl, x) is not None
    assert _eigh(dbl, x) is not None
    assert _eigvals(dbl, x) is not None
    assert _eigvalsh(dbl, x) is not None

    class DummyBackendFlat(MockTestStub):
        """Test class DummyBackendFlat."""

        def eig(self, x: BackendArg) -> BackendArg:
            """Test function eig.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `eig`.
            """
            return x

        def eigh(self, x: BackendArg) -> BackendArg:
            """Test function eigh.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `eigh`.
            """
            return x

        def eigvals(self, x: BackendArg) -> BackendArg:
            """Test function eigvals.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `eigvals`.
            """
            return x

        def eigvalsh(self, x: BackendArg) -> BackendArg:
            """Test function eigvalsh.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `eigvalsh`.
            """
            return x

    dbf = DummyBackendFlat()
    assert _eig(dbf, x) is not None
    assert _eigh(dbf, x) is not None
    assert _eigvals(dbf, x) is not None
    assert _eigvalsh(dbf, x) is not None


def test_core_math_ops_cholesky_funcs() -> None:
    """Test cholesky, cholesky_ex, cholesky_solve fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cholesky, _cholesky_ex, _cholesky_solve

    x = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        require_callable(_cholesky)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_cholesky_ex)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_cholesky_solve)(np, b, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_cholesky)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_cholesky_ex)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_cholesky_solve)(db, b, x)(db, not None)

    class DummyBackendLinalg(MockTestStub):
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def cholesky(x: BackendArg) -> BackendArg:
                """Test function cholesky.

                Args:
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `cholesky`.
                """
                return x

            @staticmethod
            def cho_solve(b: BackendArg, x: BackendArg) -> BackendArg:
                """Test function cho_solve.

                Args:
                    b: Test parameter `b`.
                    x: Test parameter `x`.

                Returns:
                    BackendArg: Computed output value from `cho_solve`.
                """
                return b

        def zeros(self, shape: BackendArg, dtype: BackendArg) -> BackendArg:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `zeros`.
            """
            return 0

    dbl = DummyBackendLinalg()
    assert _cholesky(dbl, x) is not None
    assert _cholesky_ex(dbl, x) is not None
    assert _cholesky_solve(dbl, b, x) is not None

    class DummyBackendFlat(MockTestStub):
        """Test class DummyBackendFlat."""

        def cholesky(self, x: BackendArg) -> BackendArg:
            """Test function cholesky.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `cholesky`.
            """
            return x

        def cho_solve(self, b: BackendArg, x: BackendArg) -> BackendArg:
            """Test function cho_solve.

            Args:
                b: Test parameter `b`.
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `cho_solve`.
            """
            return b

    dbf = DummyBackendFlat()
    assert _cholesky(dbf, x) is not None
    assert _cholesky_ex(dbf, x) is not None
    assert _cholesky_solve(dbf, b, x) is not None


def test_core_math_ops_banded_householder_funcs() -> None:
    """Test banded_triangular_solve, householder_product fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _banded_triangular_solve, _householder_product

    a = np.array([[0, 1], [2, 3], [4, 0]])
    b = np.array([1, 2])
    with pytest.raises(TypeError):
        require_callable(_banded_triangular_solve)(np, a, b)(np, not None)
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    tau = np.array([1.0, 1.0])

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_banded_triangular_solve)(db, a, b)(db, not None)
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(AttributeError):
            require_callable(_householder_product)(db, v, tau)(db, not None)


def test_core_math_ops_matrix_power_rank() -> None:
    """Test matrix_power, matrix_rank fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _matrix_power, _matrix_rank

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_matrix_power)(np, x, 2)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_matrix_rank)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_matrix_power)(db, x, 2)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_matrix_rank)(db, x)(db, not None)


def test_core_math_ops_norm_pinv() -> None:
    """Test norm, pinv fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _norm, _pinv

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_norm)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_pinv)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_norm)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_pinv)(db, x)(db, not None)


def test_core_math_ops_qr_slogdet() -> None:
    """Test qr, slogdet fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _qr, _slogdet

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        require_callable(_qr)(np, x)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_slogdet)(np, x)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_qr)(db, x)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_slogdet)(db, x)(db, not None)


def test_core_math_ops_solve_svd() -> None:
    """Test solve, svd fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _solve, _svd

    a = np.array([[1.0, 2.0], [3.0, 5.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        require_callable(_solve)(np, a, b)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_svd)(np, a)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_solve)(db, a, b)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_svd)(db, a)(db, not None)


def test_core_math_ops_tensorinv_solve() -> None:
    """Test tensorinv, tensorsolve fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _tensorinv, _tensorsolve

    a = np.eye(4).reshape(2, 2, 2, 2)
    b = np.eye(2)
    with pytest.raises(TypeError):
        require_callable(_tensorinv)(np, a)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_tensorsolve)(np, a, b)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_tensorinv)(db, a)(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_tensorsolve)(db, a, b)(db, not None)


def test_core_math_ops_bincount_correlate() -> None:
    """Test bincount, correlate fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bincount, _correlate

    with pytest.raises(TypeError):
        require_callable(_bincount)(np, [1, 1, 2])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_correlate)(np, [1, 2, 3], [0, 1])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_bincount)(db, [1, 1, 2])(db, not None)
    with pytest.raises(TypeError):
        require_callable(_correlate)(db, [1, 2, 3], [0, 1])(db, not None)


def test_core_math_ops_cross_cummax_cummin() -> None:
    """Test cross, cummax, cummin fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cross, _cummax, _cummin

    with pytest.raises(TypeError):
        require_callable(_cross)(np, [1, 0, 0], [0, 1, 0])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_cummax)(np, [1, 3, 2])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_cummin)(np, [2, 1, 3])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_cross)(db, [1, 0, 0], [0, 1, 0])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_cummax)(db, [1, 3, 2])(db, not None)
    with pytest.raises(AttributeError):
        require_callable(_cummin)(db, [2, 1, 3])(db, not None)

    class DummyBackendAcc(MockTestStub):
        """Test class DummyBackendAcc."""

        class maximum:
            """Test class maximum."""

            @staticmethod
            def accumulate(x: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function accumulate.

                Args:
                    x: Test parameter `x`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `accumulate`.
                """
                return x

        class minimum:
            """Test class minimum."""

            @staticmethod
            def accumulate(x: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function accumulate.

                Args:
                    x: Test parameter `x`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `accumulate`.
                """
                return x

    dba = DummyBackendAcc()
    assert _cummax(dba, [1, 3, 2]) is not None
    assert _cummin(dba, [2, 1, 3]) is not None


def test_core_math_ops_cumlogsumexp() -> None:
    """Test cumlogsumexp fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cumlogsumexp, _cumulative_logsumexp

    with pytest.raises(TypeError):
        require_callable(_cumlogsumexp)(np, [1.0, 2.0, 3.0])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_cumulative_logsumexp)(np, [1.0, 2.0, 3.0])(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_cumlogsumexp)(db, [1.0, 2.0, 3.0])(db, not None)


def test_core_math_ops_divide_multiply_no_nan() -> None:
    """Test divide_no_nan, multiply_no_nan fallbacks."""
    import warnings

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _divide_no_nan, _multiply_no_nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _divide_no_nan(np, np.array([1.0]), np.array([0.0])) is not None
        assert _multiply_no_nan(np, np.array([1.0]), np.array([np.nan])) is not None

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def divide(self, x: BackendArg, y: BackendArg) -> BackendArg:
            """Test function divide.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.

            Returns:
                BackendArg: Computed output value from `divide`.
            """
            return coerce_numeric(x) / coerce_numeric(y) if y != 0 else float("inf")

        def multiply(self, x: BackendArg, y: BackendArg) -> BackendArg:
            """Test function multiply.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.

            Returns:
                BackendArg: Computed output value from `multiply`.
            """
            return coerce_numeric(x) * coerce_numeric(y)

        def where(self, cond: BackendArg, t: BackendArg, f: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                t: Test parameter `t`.
                f: Test parameter `f`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return t if cond else f

        def isnan(self, x: BackendArg) -> BackendArg:
            """Test function isnan.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `isnan`.
            """
            import math

            return math.isnan(coerce_float(x))

    db = DummyBackend()
    assert _divide_no_nan(db, 1.0, 0.0) == 0.0
    assert _multiply_no_nan(db, float("nan"), 0.0) == 0.0
