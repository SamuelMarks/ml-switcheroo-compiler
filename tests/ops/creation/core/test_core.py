"""Tests for test_core."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode

import ml_switcheroo_compiler.ops.creation.api_basic as fb
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from tests.testing_utils import MockTensorStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _TestCreationCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_creation_coverage` split helpers."""

    buf: bytes
    original_eager: bool
    t: Tensor


def _test_creation_coverage_part_1(_state: _TestCreationCoverageState) -> None:
    """Execute part 1 of test_creation_coverage.

    Args:
        _state (_TestCreationCoverageState): Shared test state dictionary.
    """
    config.eager_mode = True

    # array
    t = fb.array([1, 2])
    assert isinstance(t, Tensor)
    t2 = fb.array([1.0, 2.0], dtype=DType("float32"))
    assert t2.dtype == DType("float32")

    # asarray
    assert fb.asarray([1, 2]) is not None
    assert fb.asarray(t).dtype == t.dtype
    assert fb.asarray(t, dtype=DType("float32")).dtype == DType("float32")

    # convert_to_tensor
    assert fb.convert_to_tensor([1, 2]) is not None

    # zeros
    z = fb.zeros((2, 2))
    assert z.shape == (2, 2)
    assert np.array_equal(ensure_ndarray(z.numpy()), np.zeros((2, 2)))
    z = fb.zeros(2)
    assert z.shape == (2,)

    # ones
    o = fb.ones((2, 2))
    assert o.shape == (2, 2)
    assert np.array_equal(ensure_ndarray(o.numpy()), np.ones((2, 2)))

    # full
    f = fb.full((2, 2), 3.0)
    assert f.shape == (2, 2)
    assert np.array_equal(ensure_ndarray(f.numpy()), np.full((2, 2), 3.0))
    _state.update({"t": t})


def _test_creation_coverage_part_2(_state: _TestCreationCoverageState) -> None:
    """Execute part 2 of test_creation_coverage.

    Args:
        _state (_TestCreationCoverageState): Shared test state dictionary.
    """
    t = _state["t"]

    class DummyFill(float):
        """Test class DummyFill."""

        data = 4.0

    f2 = fb.full((2, 2), DummyFill())
    assert np.array_equal(ensure_ndarray(f2.numpy()), np.full((2, 2), 4.0))
    zl = fb.zeros_like(t)
    assert zl.shape == t.shape
    assert np.array_equal(ensure_ndarray(zl.numpy()), np.zeros((2,)))

    # ones_like
    ol = fb.ones_like(t)
    assert ol.shape == t.shape
    assert np.array_equal(ensure_ndarray(ol.numpy()), np.ones((2,)))

    # full_like
    fl = fb.full_like(t, 5.0)
    assert fl.shape == t.shape
    assert np.array_equal(ensure_ndarray(fl.numpy()), np.full((2,), 5.0))

    # empty
    e = fb.empty((2, 2))
    assert e.shape == (2, 2)

    # empty_like
    el = fb.empty_like(t)
    assert el.shape == t.shape
    assert fb.empty_like(t, dtype=DType("float32")).dtype == DType("float32")

    # convert_to_numpy
    assert isinstance(fb.convert_to_numpy(t), np.ndarray)
    _state.update({"t": t})


def _test_creation_coverage_part_3(_state: _TestCreationCoverageState) -> None:
    """Execute part 3 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyNoNumpy(MockTensorStub):
        """Test class DummyNoNumpy."""

        data = [1, 2]

    assert isinstance(fb.convert_to_numpy(DummyNoNumpy()), np.ndarray)

    # frombuffer
    buf = b"hello"
    fb_buf = fb.frombuffer(buf, dtype=DType("int8"), count=5)
    assert fb_buf.shape == (5,)

    # tracing mode
    original_eager = config.eager_mode
    _state.update({"buf": buf, "original_eager": original_eager})


def _test_creation_coverage_part_4(_state: _TestCreationCoverageState) -> None:
    """Execute part 4 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    buf, original_eager, t = _state["buf"], _state["original_eager"], _state["t"]
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes: dict[str, LogicalNode] = {}

            def add_node(self, node: BackendArg) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", DummyGraph(), raising=False)
        import ml_switcheroo_compiler.ops.registry as registry

        def dummy_emit_creation_node(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function dummy_emit_creation_node.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `dummy_emit_creation_node`.
            """
            return Tensor(None, TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))

        registry.register_util("_emit_creation_node")(dummy_emit_creation_node)

        # Tracing node creation
        fb.array([1, 2])
        fb.zeros((2, 2))
        fb.ones((2, 2))
        fb.full((2, 2), 3.0)
        fb.zeros_like(t)
        fb.ones_like(t)
        fb.full_like(t, 5.0)
        fb.empty((2, 2))
        fb.frombuffer(buf, dtype=DType("int8"), count=5)

        class DummyNodeShape(MockTensorStub):
            """Test class DummyNodeShape."""

            data = 2

        fb._unpack_shape((DummyNodeShape(),))

    finally:
        pytest.MonkeyPatch().setattr(config, "eager_mode", original_eager, raising=False)
        global_tracing_state.is_tracing = False


def test_creation_coverage() -> None:
    """Test function test_creation_coverage."""
    _state: _TestCreationCoverageState = {}
    _test_creation_coverage_part_1(_state)
    _test_creation_coverage_part_2(_state)
    _test_creation_coverage_part_3(_state)
    _test_creation_coverage_part_4(_state)
    assert len(_state) > 0
