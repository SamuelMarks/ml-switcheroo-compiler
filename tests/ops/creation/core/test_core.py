"""Tests for test_core."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

import ml_switcheroo_compiler.ops.creation.frontend_basic as fb
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_creation_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
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
    assert cast("EagerTensorProtocol", getattr((z.numpy() == 0), "all"))()
    z = fb.zeros(2)
    assert z.shape == (2,)

    # ones
    o = fb.ones((2, 2))
    assert o.shape == (2, 2)
    assert cast("EagerTensorProtocol", getattr((o.numpy() == 1), "all"))()

    # full
    f = fb.full((2, 2), 3.0)
    assert f.shape == (2, 2)
    assert cast("EagerTensorProtocol", getattr((f.numpy() == 3.0), "all"))()
    getattr(_state, "update")({"f": f, "o": o, "t": t, "t2": t2})
    getattr(_state, "update")({"z": z})
    _state.update(locals())


def _test_creation_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]

    class DummyFill:
        """Test class DummyFill."""

        data = 4.0

    f2 = cast("Callable[..., EagerTensorProtocol]", fb.full)((2, 2), cast("Callable[..., EagerTensorProtocol]", DummyFill)())
    assert cast("EagerTensorProtocol", getattr((f2.numpy() == 4.0), "all"))()
    zl = cast("Callable[..., EagerTensorProtocol]", fb.zeros_like)(t)
    assert zl.shape == t.shape
    assert cast("EagerTensorProtocol", getattr((zl.numpy() == 0), "all"))()

    # ones_like
    ol = cast("Callable[..., EagerTensorProtocol]", fb.ones_like)(t)
    assert ol.shape == t.shape
    assert cast("EagerTensorProtocol", getattr((ol.numpy() == 1), "all"))()

    # full_like
    fl = cast("Callable[..., EagerTensorProtocol]", fb.full_like)(t, 5.0)
    assert fl.shape == t.shape
    assert cast("EagerTensorProtocol", getattr((fl.numpy() == 5.0), "all"))()

    # empty
    e = fb.empty((2, 2))
    assert e.shape == (2, 2)

    # empty_like
    el = cast("Callable[..., EagerTensorProtocol]", fb.empty_like)(t)
    assert el.shape == t.shape
    assert cast("Callable[..., EagerTensorProtocol]", fb.empty_like)(t, dtype=cast("Callable[..., EagerTensorProtocol]", DType)("float32")).dtype == cast("Callable[..., EagerTensorProtocol]", DType)("float32")

    # convert_to_numpy
    assert isinstance(cast("Callable[..., EagerTensorProtocol]", fb.convert_to_numpy)(t), np.ndarray)
    getattr(_state, "update")({"DummyFill": DummyFill, "e": e, "el": el, "f2": f2})
    getattr(_state, "update")({"fl": fl, "ol": ol, "zl": zl})
    _state.update(locals())


def _test_creation_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyNoNumpy:
        """Test class DummyNoNumpy."""

        data = [1, 2]

    assert isinstance(cast("Callable[..., EagerTensorProtocol]", fb.convert_to_numpy)(cast("Callable[..., EagerTensorProtocol]", DummyNoNumpy)()), np.ndarray)

    # frombuffer
    buf = b"hello"
    fb_buf = fb.frombuffer(buf, dtype=DType("int8"), count=5)
    assert fb_buf.shape == (5,)

    # tracing mode
    original_eager = config.eager_mode
    getattr(_state, "update")({"DummyNoNumpy": DummyNoNumpy, "buf": buf, "fb_buf": fb_buf, "original_eager": original_eager})
    _state.update(locals())


def _test_creation_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
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
            nodes = cast("dict[str, EagerTensorProtocol]", {})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node."""
                pass

        setattr(global_tracing_state, "active_graph", DummyGraph())
        import ml_switcheroo_compiler.ops.registry as registry

        def dummy_emit_creation_node(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function dummy_emit_creation_node."""
            return Tensor(None, TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))

        registry.register_util("_emit_creation_node")(dummy_emit_creation_node)

        # Tracing node creation
        fb.array([1, 2])
        fb.zeros((2, 2))
        fb.ones((2, 2))
        fb.full((2, 2), 3.0)
        cast("Callable[..., EagerTensorProtocol]", fb.zeros_like)(t)
        cast("Callable[..., EagerTensorProtocol]", fb.ones_like)(t)
        cast("Callable[..., EagerTensorProtocol]", fb.full_like)(t, 5.0)
        fb.empty((2, 2))
        fb.frombuffer(buf, dtype=DType("int8"), count=5)

        class DummyNodeShape:
            """Test class DummyNodeShape."""

            data = 2

        cast("Callable[..., EagerTensorProtocol]", fb._unpack_shape)((cast("Callable[..., EagerTensorProtocol]", DummyNodeShape)(),))

    finally:
        setattr(config, "eager_mode", original_eager)
        global_tracing_state.is_tracing = False


def test_creation_coverage() -> None:
    """Test function test_creation_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_creation_coverage_part_1(_state)
    _test_creation_coverage_part_2(_state)
    _test_creation_coverage_part_3(_state)
    _test_creation_coverage_part_4(_state)
    assert isinstance(_state, dict)
