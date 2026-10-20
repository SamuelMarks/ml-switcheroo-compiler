"""Tests for test_utils."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_emit_shape_node(mocker: EagerTensorProtocol) -> None:
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

    class FakeDtype1:
        """Test class FakeDtype1."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.value = "int32"

    typing.cast("EagerCallableProtocol", _emit_shape_node)("Op", [t], {}, (2, 3), typing.cast("EagerCallableProtocol", FakeDtype1)())

    class FakeDtype2:
        """Test class FakeDtype2."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "int64"

        def __str__(self) -> str:
            """Test function __str__."""
            return "int64"

    typing.cast("EagerCallableProtocol", _emit_shape_node)("Op", [t], {}, (2, 3), typing.cast("EagerCallableProtocol", FakeDtype2)())

    class FakeDtype3:
        """Test class FakeDtype3."""

        pass

    typing.cast("EagerCallableProtocol", _emit_shape_node)("Op", [t], {}, (2, 3), typing.cast("EagerCallableProtocol", FakeDtype3)())

    class FakeNumpyDtype:
        """Test class FakeNumpyDtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "float64"

    FakeNumpyDtype.__name__ = "dtype"

    class dtype:
        """Test class dtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "float64"

    res = typing.cast("EagerCallableProtocol", _emit_shape_node)("Op", [t], {}, (2, 3), typing.cast("EagerCallableProtocol", dtype)())
    assert res is not None


def test_compute_reduction_shape() -> None:
    """Test function test_compute_reduction_shape."""
    from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

    assert compute_reduction_shape((2, 3, 4), (1,), True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), False) == (2, 4)
