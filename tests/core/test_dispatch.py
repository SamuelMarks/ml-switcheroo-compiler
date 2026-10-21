"""Test core dispatch."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dispatch import dispatch
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import global_tracing_state
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_dispatch_eager_unsupported(monkeypatch: EagerTensorProtocol) -> None:
    """Test dispatch fallback in eager mode when unsupported.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    config.eager_mode = True

    # Mock backend to simulate an unsupported function
    mock_backend = MagicMock()
    mock_backend.module = MagicMock()
    delattr(mock_backend.module, "non_existent_module")

    monkeypatch.setattr("ml_switcheroo_compiler.core.dispatch.get_active_backend", lambda: mock_backend)

    with pytest.raises(ValueError, match="is not supported in the active backend."):
        dispatch("non_existent_module", "non_existent_func")
    config.eager_mode = False


def test_dispatch_tracing() -> None:
    """Test dispatch in tracing mode."""
    config.eager_mode = False

    class MockTensor:
        """Test class MockTensor."""

        dtype = "float32"

    global_tracing_state.start_tracing()
    try:
        from ml_switcheroo_compiler.ops.base import OpDef, register_op

        @register_op("TestDispatchTracing")
        class TestDispatchTracing(OpDef):
            """Test class TestDispatchTracing."""

            op_name = "TestDispatchTracing"

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function infer_shape.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                raise ValueError("Simulate shape inference failure")

        t = Tensor(None, TensorConfig((2, 2), "float32", "cpu"))
        out = dispatch("test", "TestDispatchTracing", t)
        assert as_tensor(getattr(out, "shape")) == ()
        assert as_tensor(getattr(out, "dtype")).value == "float32"
    finally:
        global_tracing_state.stop_tracing()
