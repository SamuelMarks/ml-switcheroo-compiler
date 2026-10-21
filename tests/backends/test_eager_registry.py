"""Test module."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerOpRegistry
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_eager_registry() -> None:
    """Test function test_eager_registry."""
    r = EagerOpRegistry()
    assert r.get("foo") is None

    @r.register("foo")
    def foo(x: EagerValue) -> EagerValue:
        """Test function foo.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) + as_tensor(1)

    assert r.get("foo") is foo

    assert r.dispatch("foo", 1) == 2

    with pytest.raises(ValueError):
        r.dispatch("bar_unique_not_existing", 1)


def test_eager_registry_fallback() -> None:
    """Test function test_eager_registry_fallback."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    r = EagerOpRegistry()

    @global_eager_registry.register("global_foo_test")
    def gfoo(x: EagerValue) -> EagerValue:
        """Test function gfoo.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) + as_tensor(2)

    assert r.dispatch("global_foo_test", 3) == 5


def test_global_eager_registry_fallback_missing() -> None:
    """Test function test_global_eager_registry_fallback_missing."""
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    with pytest.raises(ValueError):
        global_eager_registry.dispatch("op_does_not_exist_ever_123")


def test_decode_image_eager() -> None:
    """Test function test_decode_image_eager."""
    import struct

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image_camel

    # Test empty args
    assert as_callable(len)(as_callable(_np_decode_image_camel)(None)) == 0

    # Test not bytes
    with pytest.raises(RuntimeError, match="Expected bytes"):
        _np_decode_image_camel(as_mod(None), np.array([1, 2, 3]))

    # Construct valid 10x10 RGB PNG header
    png_bytes = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 10, 10, 8, 2, 0, 0, 0)

    # Default channels
    res = _np_decode_image_camel(as_mod(None), png_bytes)
    assert as_tensor(getattr(res, "shape")) == (10, 10, 3)

    # Channels = 1
    res1 = _np_decode_image_camel(as_mod(None), png_bytes, channels=1)
    assert as_tensor(getattr(res1, "shape")) == (10, 10, 1)

    # Channels = 4
    res4 = _np_decode_image_camel(as_mod(None), png_bytes, channels=4)
    assert as_tensor(getattr(res4, "shape")) == (10, 10, 4)


def test_encode_image_eager() -> None:
    """Test function test_encode_image_eager."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_encode_image_camel

    # Test empty args
    res = _np_encode_image_camel(as_mod(None))
    assert as_tensor(getattr(res, "item"))() == b""

    # Shape with 1 channel
    arr = np.ones((10, 10, 1), dtype=np.uint8)
    res = _np_encode_image_camel(as_mod(None), arr)
    assert isinstance(res, np.ndarray)

    # Non-uint8 dtype
    arr2 = np.ones((10, 10, 3), dtype=np.float32)
    res2 = _np_encode_image_camel(as_mod(None), arr2)
    assert isinstance(res2, np.ndarray)

    # Trigger exception
    with patch("struct.pack", side_effect=Exception("mock error")):
        with pytest.raises(RuntimeError):
            _np_encode_image_camel(as_mod(None), arr2)


def test_load_vision_formats_missing() -> None:
    """Test function test_load_vision_formats_missing."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}


def test_dynamic_fallback() -> None:
    """Test function test_dynamic_fallback."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    class DummyObj:
        """Test class DummyObj."""

        def my_op(self, *args: EagerValue) -> EagerValue:
            """Test function my_op.

            Args:
                args: Test parameter `args`.
            """
            return 42

    assert as_callable(global_eager_registry.dispatch)("My_Op", as_callable(DummyObj)()) == 42

    class DummyBackend:
        """Test class DummyBackend."""

        def my_backend_op(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function my_backend_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return 43

        def subtract(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function subtract.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return 44

    assert as_callable(global_eager_registry.dispatch)("My_Backend_Op", 1, backend_module=as_callable(DummyBackend)()) == 43
    assert as_callable(global_eager_registry.dispatch)("sub", as_callable(DummyBackend)(), backend_module=as_callable(DummyBackend)()) == 44

    with pytest.raises(ValueError, match="not found in registry and no universal fallback"):
        as_callable(global_eager_registry.dispatch)("missing_op_on_backend", backend_module=as_callable(DummyBackend)())


def test_custom_vjp_and_tuple() -> None:
    """Test function test_custom_vjp_and_tuple."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    assert global_eager_registry.dispatch("CustomVJP", None, 1) == 1
    assert global_eager_registry.dispatch("CustomVJP", None, 1, 2) == (1, 2)

    def my_bwd(res: EagerValue, *args: EagerValue) -> EagerValue:
        """Test function my_bwd.

        Args:
            res: Test parameter `res`.
            args: Test parameter `args`.
        """
        return as_callable(sum)(args)

    assert global_eager_registry.dispatch("ProcessCustomVJPCall", None, 1, 2, bwd_fn=my_bwd) == 3
    assert global_eager_registry.dispatch("ProcessCustomVJPCall", None, 1, 2, bwd_fn="not callable") is None

    assert global_eager_registry.dispatch("TupleGetItem", None, (10, 20), index=1) == 20
    assert global_eager_registry.dispatch("TupleGetItem", None, "not a tuple", index=0) is None
