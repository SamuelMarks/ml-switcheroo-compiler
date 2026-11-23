"""Test module."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerOpRegistry, coerce_bytes
from tests.testing_utils import (
    MockTestStub,
    adapt_backend_module,
    coerce_numeric,
    coerce_operand_list,
    extract_test_shape,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_eager_registry() -> None:
    """Test function test_eager_registry."""
    r = EagerOpRegistry()
    assert r.get("foo") is None

    @r.register("foo")
    def foo(x: BackendArg) -> BackendArg:
        """Test function foo.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `foo`.
        """
        return coerce_numeric(x) + 1

    assert r.get("foo") is foo

    assert r.dispatch("foo", 1) == 2

    with pytest.raises(ValueError):
        r.dispatch("bar_unique_not_existing", 1)


def test_eager_registry_fallback() -> None:
    """Test function test_eager_registry_fallback."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    r = EagerOpRegistry()

    @global_eager_registry.register("global_foo_test")
    def gfoo(x: BackendArg) -> BackendArg:
        """Test function gfoo.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `gfoo`.
        """
        return coerce_numeric(x) + 2

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

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_image_camel

    # Test empty args
    assert len(coerce_operand_list(_np_decode_image_camel(None))) == 0

    # Test not bytes
    with pytest.raises(RuntimeError, match="Expected bytes"):
        _np_decode_image_camel(None, np.array([1, 2, 3]))

    # Construct valid 10x10 RGB PNG header
    png_bytes = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 10, 10, 8, 2, 0, 0, 0)

    # Default channels
    res = _np_decode_image_camel(None, png_bytes)
    assert extract_test_shape(res) == (10, 10, 3)

    # Channels = 1
    res1 = _np_decode_image_camel(None, png_bytes, channels=1)
    assert extract_test_shape(res1) == (10, 10, 1)

    # Channels = 4
    res4 = _np_decode_image_camel(None, png_bytes, channels=4)
    assert extract_test_shape(res4) == (10, 10, 4)


def test_encode_image_eager() -> None:
    """Test function test_encode_image_eager."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_encode_image_camel
    from ml_switcheroo_compiler.core.shape import OpValue

    # Test empty args
    res = _np_encode_image_camel(None)
    res_item: OpValue = np.asarray(res).tolist()
    assert res_item == b""

    # Shape with 1 channel
    arr = np.ones((10, 10, 1), dtype=np.uint8)
    res = _np_encode_image_camel(None, arr)
    assert isinstance(res, np.ndarray)

    # Non-uint8 dtype
    arr2 = np.ones((10, 10, 3), dtype=np.float32)
    res2 = _np_encode_image_camel(None, arr2)
    assert isinstance(res2, np.ndarray)

    # Trigger exception
    with patch("struct.pack", side_effect=RuntimeError("mock error")):
        with pytest.raises(RuntimeError):
            _np_encode_image_camel(None, arr2)


def test_load_vision_formats_missing() -> None:
    """Test function test_load_vision_formats_missing."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}


def test_dynamic_fallback() -> None:
    """Test function test_dynamic_fallback."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    class DummyObj(MockTestStub):
        """Test class DummyObj."""

        def my_op(self, *args: BackendArg) -> BackendArg:
            """Test function my_op.

            Args:
                args: Test parameter `args`.

            Returns:
                BackendArg: Computed output value from `my_op`.
            """
            return 42

    assert global_eager_registry.dispatch("My_Op", DummyObj()) == 42

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def my_backend_op(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function my_backend_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `my_backend_op`.
            """
            return 43

        def subtract(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function subtract.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `subtract`.
            """
            return 44

    assert global_eager_registry.dispatch("My_Backend_Op", 1, backend_module=DummyBackend()) == 43
    assert global_eager_registry.dispatch("sub", DummyBackend(), backend_module=DummyBackend()) == 44

    with pytest.raises(ValueError, match="not found in registry and no universal fallback"):
        global_eager_registry.dispatch("missing_op_on_backend", backend_module=DummyBackend())


def test_custom_vjp_and_tuple() -> None:
    """Test function test_custom_vjp_and_tuple."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    assert global_eager_registry.dispatch("CustomVJP", None, 1) == 1
    assert global_eager_registry.dispatch("CustomVJP", None, 1, 2) == (1, 2)

    def my_bwd(res: BackendArg, *args: BackendArg) -> BackendArg:
        """Test function my_bwd.

        Args:
            res (BackendArg): Test parameter `res`.
            *args (BackendArg): Test parameter `args`.

        Returns:
            BackendArg: Computed output value from `my_bwd`.
        """
        del res
        return sum([coerce_numeric(x) for x in args])

    assert global_eager_registry.dispatch("ProcessCustomVJPCall", None, 1, 2, bwd_fn=my_bwd) == 3
    assert global_eager_registry.dispatch("ProcessCustomVJPCall", None, 1, 2, bwd_fn="not callable") is None

    assert global_eager_registry.dispatch("TupleGetItem", None, (10, 20), index=1) == 20
    assert global_eager_registry.dispatch("TupleGetItem", None, "not a tuple", index=0) is None


def test_eager_registry_narrowing_and_schema_hooks() -> None:
    """Verify runtime narrowing utilities, Pydantic validator hooks, and split protocol exports."""
    from ml_switcheroo_compiler.backends.eager_registry import (
        CORE_EAGER_OP_TABLE,
        BackendArrayProtocol,
        BackendContextManagerProtocol,
        BackendRNGProtocol,
        EagerOpCallable,
        EagerTensorProtocol,
        coerce_bool,
        coerce_float,
        coerce_float_list,
        coerce_int,
        coerce_int_list,
        coerce_int_or_tuple,
        coerce_operand_list,
        coerce_opt_callable,
        coerce_opt_int,
        coerce_opt_int_seq,
        coerce_opt_str,
        coerce_shape,
        coerce_str,
        unpack_pair,
        unpack_quad,
        unpack_triple,
    )
    from ml_switcheroo_compiler.backends.eager_registry.types_base import _passthrough_tensor_val

    assert BackendArrayProtocol is not None
    assert BackendContextManagerProtocol is not None
    assert BackendRNGProtocol is not None
    assert EagerOpCallable is not None
    assert "CustomVJP" in CORE_EAGER_OP_TABLE

    assert _passthrough_tensor_val(7) == 7
    dummy_handler = adapt_backend_module(np).asarray([0.0])
    schema = EagerTensorProtocol.__get_pydantic_core_schema__(int, dummy_handler)
    assert isinstance(schema, dict) and schema["type"] == "function-plain"

    assert coerce_int(5) == 5
    assert coerce_float(2.5) == 2.5
    assert coerce_shape((2, 3)) == (2, 3)
    assert coerce_int_list([1, 2]) == [1, 2]
    assert coerce_str("abc") == "abc"
    assert coerce_opt_str(None) is None
    assert coerce_opt_str("xyz") == "xyz"
    assert coerce_opt_int(None) is None
    assert coerce_opt_int(9) == 9
    assert coerce_bool(True) is True
    assert coerce_bytes(b"raw") == b"raw"
    assert coerce_bytes([65, 66]) == b"AB"
    assert coerce_float_list([1.0, 2.0]) == [1.0, 2.0]
    assert coerce_operand_list([1, "a"]) == [1, "a"]

    pair = unpack_pair((1, 2))
    assert len(pair) == 2
    triple = unpack_triple((1, 2, 3))
    assert len(triple) == 3
    quad = unpack_quad((1, 2, 3, 4))
    assert len(quad) == 4

    assert coerce_opt_int_seq(None) is None
    assert coerce_opt_int_seq((1, 2)) == (1, 2)
    assert coerce_int_or_tuple(3) == 3
    assert coerce_int_or_tuple((3, 4)) == (3, 4)
    assert coerce_opt_callable(None) is None
    fn_ref = coerce_opt_callable(len)
    assert fn_ref is not None


def test_coerce_int_and_float_mock_and_module_guards() -> None:
    """Verify coerce_int and coerce_float prevent NonCallableMock .item auto-vivification and bound unwrapping."""
    import types
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.backends.eager_registry import coerce_float, coerce_int

    bare_mock = MagicMock()
    assert coerce_int(bare_mock) == 1
    assert coerce_float(bare_mock) == 1.0
    assert "item" not in bare_mock._mock_children

    item_mock = MagicMock()
    item_mock.item.return_value = 7
    assert coerce_int(item_mock) == 7
    assert coerce_float(item_mock) == 7.0

    attr_item_mock = MagicMock()
    attr_item_mock.item = lambda: 9
    assert coerce_int(attr_item_mock) == 9
    assert coerce_float(attr_item_mock) == 9.0

    deleted_item_mock = MagicMock()
    _ = deleted_item_mock.item
    del deleted_item_mock.item
    assert coerce_int(deleted_item_mock) == 1
    assert coerce_float(deleted_item_mock) == 1.0

    nested_mock = MagicMock()
    nested_mock.item.return_value = MagicMock()
    assert coerce_int(nested_mock) == 1
    assert coerce_float(nested_mock) == 1.0

    scalar_arr = np.array(5)
    assert coerce_int(scalar_arr) == 5
    assert coerce_float(scalar_arr) == 5.0

    multi_arr = np.array([1, 2])
    with pytest.raises(TypeError):
        coerce_int(multi_arr)
    with pytest.raises(TypeError):
        coerce_float(multi_arr)

    with pytest.raises(TypeError):
        coerce_int(types)
    with pytest.raises(TypeError):
        coerce_float(types)
