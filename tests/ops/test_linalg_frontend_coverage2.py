"""Module docstring."""

from unittest.mock import MagicMock

import numpy as np

import ml_switcheroo_compiler.ops.linalg.utils as lf
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import TriangularSolveOptions
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

    monkeypatch.setattr(lf, "global_tracing_state", MockTracer())

    try:

        class MockData:
            """Docstring."""

            id = "test_id"

        t = Tensor(MockData(), TensorConfig((2, 2), DType.Float32, Device("cpu")))
        res = pinv(t)
        assert res is not None
    finally:
        config.eager_mode = True


def test_solve_triangular() -> None:
    """Docstring."""
    a = Tensor(np.array([[3, 0, 0], [2, 1, 0], [1, 0, 1]]), TensorConfig((3, 3), "float32", "cpu"))
    b = Tensor(np.array([4, 2, 4]), TensorConfig((3,), "float32", "cpu"))

    res = solve_triangular(a, b, TriangularSolveOptions(lower=True))
    assert res.shape == (3,)


def test_lu() -> None:
    """Docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    p, l_, u = lu(a)
    assert p.shape == (2, 2)


def test_lu_factor() -> None:
    """Docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    lu_arr, piv = lu_factor(a)
    assert lu_arr.shape == (2, 2)
