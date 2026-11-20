"""Tests for test_utils."""

from __future__ import annotations

import typing

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_emit_shape_node(mocker: MockerFixture) -> None:
    """Test function test_emit_shape_node.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils.global_tracing_state.is_tracing", True)
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    _emit_shape_node("Op", [t], {}, (2, 3), DType.Float32)

    class FakeDtype1(MockTestStub):
        """Test class FakeDtype1."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.value = "int32"

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype1())

    class FakeDtype2(MockTestStub):
        """Test class FakeDtype2."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "int64"

        def __str__(self) -> str:
            """Test function __str__.

            Returns:
                str: Computed output value from `__str__`.
            """
            return "int64"

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype2())

    class FakeDtype3(MockTestStub):
        """Test class FakeDtype3."""

        pass

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype3())

    class FakeNumpyDtype:
        """Test class FakeNumpyDtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "float64"

    FakeNumpyDtype.__name__ = "dtype"

    class dtype(MockTestStub):
        """Test class dtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "float64"

    res = _emit_shape_node("Op", [t], {}, (2, 3), dtype())
    assert tuple(res.shape) == (2, 3)


def test_compute_reduction_shape() -> None:
    """Test function test_compute_reduction_shape."""
    from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

    assert compute_reduction_shape((2, 3, 4), (1,), True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), False) == (2, 4)
