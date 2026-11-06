"""Tests for test_core."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.ops.creation.frontend_basic as fb
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
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
    assert ensure_eager_tensor(ensure_eager_tensor(z.numpy() == 0).all)()
    z = fb.zeros(2)
    assert z.shape == (2,)

    # ones
    o = fb.ones((2, 2))
    assert o.shape == (2, 2)
    assert ensure_eager_tensor(ensure_eager_tensor(o.numpy() == 1).all)()

    # full
    f = fb.full((2, 2), 3.0)
    assert f.shape == (2, 2)
    assert ensure_eager_tensor(ensure_eager_tensor(f.numpy() == 3.0).all)()
    _state.update(ensure_tensor_dict({"f": f, "o": o, "t": t, "t2": t2}))
    _state.update(ensure_tensor_dict({"z": z}))


def _test_creation_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]

    class DummyFill:
        """Test class DummyFill."""

        data = 4.0

    f2 = ensure_callable(fb.full)((2, 2), ensure_callable(DummyFill)())
    assert ensure_eager_tensor(ensure_eager_tensor(f2.numpy() == 4.0).all)()
    zl = ensure_callable(fb.zeros_like)(t)
    assert zl.shape == t.shape
    assert ensure_eager_tensor(ensure_eager_tensor(zl.numpy() == 0).all)()

    # ones_like
    ol = ensure_callable(fb.ones_like)(t)
    assert ol.shape == t.shape
    assert ensure_eager_tensor(ensure_eager_tensor(ol.numpy() == 1).all)()

    # full_like
    fl = ensure_callable(fb.full_like)(t, 5.0)
    assert fl.shape == t.shape
    assert ensure_eager_tensor(ensure_eager_tensor(fl.numpy() == 5.0).all)()

    # empty
    e = fb.empty((2, 2))
    assert e.shape == (2, 2)

    # empty_like
    el = ensure_callable(fb.empty_like)(t)
    assert el.shape == t.shape
    assert ensure_callable(fb.empty_like)(t, dtype=ensure_callable(DType)("float32")).dtype == ensure_callable(DType)("float32")

    # convert_to_numpy
    assert isinstance(ensure_callable(fb.convert_to_numpy)(t), np.ndarray)
    _state.update(ensure_tensor_dict({"DummyFill": DummyFill, "e": e, "el": el, "f2": f2}))
    _state.update(ensure_tensor_dict({"fl": fl, "ol": ol, "zl": zl}))
    _state.update(ensure_tensor_dict({"t": t}))


def _test_creation_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_creation_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyNoNumpy:
        """Test class DummyNoNumpy."""

        data = [1, 2]

    assert isinstance(ensure_callable(fb.convert_to_numpy)(ensure_callable(DummyNoNumpy)()), np.ndarray)

    # frombuffer
    buf = b"hello"
    fb_buf = fb.frombuffer(buf, dtype=DType("int8"), count=5)
    assert fb_buf.shape == (5,)

    # tracing mode
    original_eager = config.eager_mode
    _state.update(ensure_tensor_dict({"DummyNoNumpy": DummyNoNumpy, "buf": buf, "fb_buf": fb_buf, "original_eager": original_eager}))


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
            nodes = ensure_tensor_dict({})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
        import ml_switcheroo_compiler.ops.registry as registry

        def dummy_emit_creation_node(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function dummy_emit_creation_node.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return Tensor(None, TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))

        registry.register_util("_emit_creation_node")(dummy_emit_creation_node)

        # Tracing node creation
        fb.array([1, 2])
        fb.zeros((2, 2))
        fb.ones((2, 2))
        fb.full((2, 2), 3.0)
        ensure_callable(fb.zeros_like)(t)
        ensure_callable(fb.ones_like)(t)
        ensure_callable(fb.full_like)(t, 5.0)
        fb.empty((2, 2))
        fb.frombuffer(buf, dtype=DType("int8"), count=5)

        class DummyNodeShape:
            """Test class DummyNodeShape."""

            data = 2

        ensure_callable(fb._unpack_shape)((ensure_callable(DummyNodeShape)(),))

    finally:
        ensure_mutable(config).eager_mode = ensure_eager_tensor(original_eager)
        global_tracing_state.is_tracing = False


def test_creation_coverage() -> None:
    """Test function test_creation_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_creation_coverage_part_1(_state)
    _test_creation_coverage_part_2(_state)
    _test_creation_coverage_part_3(_state)
    _test_creation_coverage_part_4(_state)
    assert len(_state) > 0
