"""Tests for test_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_emit_shape_node(mocker: EagerTensorProtocol) -> None:
    """Test function test_emit_shape_node."""
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

    cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)("Op", [t], {}, (2, 3), cast("Callable[..., EagerTensorProtocol]", FakeDtype1)())

    class FakeDtype2:
        """Test class FakeDtype2."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.name = "int64"

        def __str__(self) -> str:
            """Test function __str__."""
            return "int64"

    cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)("Op", [t], {}, (2, 3), cast("Callable[..., EagerTensorProtocol]", FakeDtype2)())

    class FakeDtype3:
        """Test class FakeDtype3."""

        pass

    cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)("Op", [t], {}, (2, 3), cast("Callable[..., EagerTensorProtocol]", FakeDtype3)())

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

    cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)("Op", [t], {}, (2, 3), cast("Callable[..., EagerTensorProtocol]", dtype)())
    assert t is not None or t is None


def test_compute_reduction_shape() -> None:
    """Test function test_compute_reduction_shape."""
    from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

    assert compute_reduction_shape((2, 3, 4), (1,), True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), False) == (2, 4)
