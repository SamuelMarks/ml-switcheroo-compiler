"""Tests for test_checkpointing."""

from __future__ import annotations

import typing

import pytest

from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_checkpointing() -> None:
    """Test function test_checkpointing."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.checkpointing import checkpoint
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # Eager mode
    def dummy(x: Tensor | int) -> Tensor | int:
        """Test function dummy.

        Args:
            x (Tensor | int): Test parameter `x`.

        Returns:
            Tensor | int: Computed output value from `dummy`.
        """
        return x + 1

    cp = checkpoint(dummy)
    assert cp(1) == 2

    # Traced mode
    config.eager_mode = False
    global_tracing_state.start_tracing()

    class FakeData:
        """Test class FakeData."""

        def __init__(self, id: BackendArg) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
            """
            self.id = id

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    pytest.MonkeyPatch().setattr(t1, "_data", FakeData("n1"), raising=False)

    t2 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    pytest.MonkeyPatch().setattr(t2, "_data", FakeData("n2"), raising=False)

    cp(t1)

    # with dtype in attr
    def dummy_dtype(x: BackendArg) -> BackendArg:
        """Test function dummy_dtype.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `dummy_dtype`.
        """
        return x

    # Test fallback
    from unittest.mock import patch

    from tests.testing_utils import MockGraphStub, MockNodeStub

    fake_block = MockGraphStub(
        outputs=["out"],
        nodes={
            "out": MockNodeStub(id="out", op_type="Output", inputs=["real"]),
            "real": MockNodeStub(id="real", op_type="Relu", shape_metadata=(2,), attributes={"dtype": "int32"}),
        },
    )

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value=fake_block):
        res2 = cp(t1)
        assert extract_test_shape(res2) == (2,)

    fake_block2 = MockGraphStub(
        outputs=["out"],
        nodes={
            "out": MockNodeStub(id="out", op_type="Output", inputs=["real"]),
            "real": MockNodeStub(id="real", op_type="Relu", shape_metadata=(2,), attributes={}),
        },
    )

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value=fake_block2):
        res3 = cp()
        assert extract_test_shape(res3) == (2,)

    config.eager_mode = True
    global_tracing_state.stop_tracing()


def test_remat_and_recompute_grad() -> None:
    """Test function test_remat_and_recompute_grad."""
    from ml_switcheroo_compiler.grad.checkpointing import recompute_grad, remat

    def dummy(x: BackendArg) -> BackendArg:
        """Test function dummy.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `dummy`.
        """
        return x

    # 106, 118
    assert remat(dummy)(1) == 1
    assert recompute_grad(dummy)(2) == 2
