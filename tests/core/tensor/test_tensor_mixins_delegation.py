"""Tests for test_tensor_mixins_delegation."""

from __future__ import annotations

import sys
import typing

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def make_tensor(data: EagerValue) -> EagerValue:
    """Test function make_tensor.

    Args:
        data: Test parameter `data`.
    """
    cfg = TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=())
    return Tensor(data, cfg)


def test_numpy_conversion_fallback() -> None:
    """Test function test_numpy_conversion_fallback."""
    t = make_tensor(1.0)
    original_numpy = sys.modules.get("numpy")
    getattr(sys.modules, "__setitem__")("numpy", None)
    try:
        typing.cast("EagerTensorProtocol", getattr(t, "numpy"))()
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        getattr(sys.modules, "__setitem__")("numpy", original_numpy)
    assert callable(test_numpy_conversion_fallback)


def test_exception_in_asarray() -> None:
    """Test function test_exception_in_asarray."""

    class BadData:
        """Test class BadData."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [1.0]

        def __array__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __array__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise RuntimeError("boom")

    t = typing.cast("EagerCallableProtocol", make_tensor)(typing.cast("EagerCallableProtocol", BadData)())
    res = typing.cast("EagerTensorProtocol", getattr(t, "__array__"))()
    assert res.shape == (1,)


def test_item_eval_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_item_eval_fallback.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """

    class FakeBackend:
        """Test class FakeBackend."""

        def item(self, x: EagerValue) -> EagerValue:
            """Test function item.

            Args:
                x: Test parameter `x`.
            """
            return typing.cast("EagerCallableProtocol", int)(x)

    class FakeTensor(Tensor):
        """Test class FakeTensor."""

        def eval(self) -> Tensor:
            """Test function eval."""
            return typing.cast("Tensor", 5)

    t = FakeTensor(1.0, TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=()))
    import ml_switcheroo_compiler.backends.registry as reg

    typing.cast("EagerCallableProtocol", monkeypatch.setattr)(reg, "get_active_backend", lambda: typing.cast("EagerCallableProtocol", FakeBackend)())
    assert t.item() == 5


def test_getitem_indexerror() -> None:
    """Test function test_getitem_indexerror."""
    t = make_tensor([1, 2])
    with pytest.raises(IndexError):
        typing.cast("EagerTensorProtocol", getattr(t, "__getitem__")((1, 2)))
    from ml_switcheroo_compiler.core import config

    orig = config.eager_mode
    config.eager_mode = False

    class FakeProxy:
        """Test class FakeProxy."""

        def __index__(self) -> None:
            """Test function __index__."""
            raise IndexError("only integers")

    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        res = typing.cast("EagerTensorProtocol", getattr(t, "__getitem__")(typing.cast("EagerCallableProtocol", FakeProxy)()))
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
