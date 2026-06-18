"""Module docstring."""

from unittest.mock import MagicMock

import numpy as np

import ml_switcheroo_compiler.ops.linalg.frontend as lf
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import lu, lu_factor, pinv, solve_triangular


def test_pinv_lazy(monkeypatch: object) -> None:
    """Docstring."""
    config.eager_mode = False

    class MockTracer:
        """Docstring."""

        is_tracing = True
        graph = MagicMock()

        def add_node(self, *args: object, **kwargs: object) -> str:
            """Docstring."""
            return "n1"

    monkeypatch.setattr(lf, "_tracer", MockTracer())

    try:

        class MockData:
            """Docstring."""

            id = "test_id"

        t = Tensor(MockData(), shape=(2, 2), dtype=DType.Float32, device=Device("cpu"))
        res = pinv(t)
        assert res is not None
    finally:
        config.eager_mode = True


def test_solve_triangular(monkeypatch: object) -> None:
    """Docstring."""
    import sys

    scipy = MagicMock()
    scipy.linalg.solve_triangular.return_value = "solved"
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.linalg", scipy.linalg)

    a = np.array([[3, 0, 0], [2, 1, 0], [1, 0, 1]])
    b = np.array([4, 2, 4])
    from ml_switcheroo_compiler.ops.configs import TriangularSolveOptions

    res = solve_triangular(a, b, TriangularSolveOptions(lower=True))
    assert res == "solved"


def test_lu(monkeypatch: object) -> None:
    """Docstring."""
    import sys

    scipy = MagicMock()
    scipy.linalg.lu.return_value = ("p", "l", "u")
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.linalg", scipy.linalg)

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, l_, u = lu(a)
    assert p == "p"


def test_lu_factor(monkeypatch: object) -> None:
    """Docstring."""
    import sys

    scipy = MagicMock()
    scipy.linalg.lu_factor.return_value = ("lu", "piv")
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.linalg", scipy.linalg)

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    lu_arr, piv = lu_factor(a)
    assert lu_arr == "lu"
