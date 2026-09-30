"""Tests for test_base64_io_coverage."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.io.base64_io as b64_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_base64_io_coverage() -> None:
    """Verify 100% line and branch coverage for base64_io operations."""
    # Test _eager_base64 helper
    # 1. None element
    assert b64_mod._eager_base64("encode", None) == b""
    # 2. String input, encode with pad=False and pad=True
    raw_str = "hello world"
    encoded_unpadded = b64_mod._eager_base64("encode", raw_str, pad=False)
    encoded_padded = b64_mod._eager_base64("encode", raw_str, pad=True)
    assert not encoded_unpadded.endswith(b"=")
    assert encoded_padded == base64.b64encode(raw_str.encode("utf-8"))

    # 3. Decode bytes
    decoded = b64_mod._eager_base64("decode", encoded_padded)
    assert decoded == b"hello world"

    # 4. List / tuple sequence
    batch_enc = b64_mod._eager_base64("encode", ["a", "b"], pad=True)
    assert isinstance(batch_enc, list)
    assert len(batch_enc) == 2

    orig_eager = config.eager_mode
    try:
        # Eager mode encode and decode
        config.eager_mode = True
        dummy_t = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, Device("cpu")))
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = "b64_backend_res"

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
            assert b64_mod.encode_base64(dummy_t, pad=True, name="op1") == "b64_backend_res"
            mock_backend.execute_op.assert_called_with("EncodeBase64", dummy_t, pad=True, name="op1")

            assert b64_mod.decode_base64(dummy_t, name="op2") == "b64_backend_res"
            mock_backend.execute_op.assert_called_with("DecodeBase64", dummy_t, name="op2")

        # Non-eager mode encode and decode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", side_effect=lambda op, *args, **kwargs: f"emitted_{op}"):
            assert b64_mod.encode_base64(dummy_t) == "emitted_EncodeBase64"
            assert b64_mod.decode_base64(dummy_t) == "emitted_DecodeBase64"

        # OpDefs infer_shape
        enc_op = b64_mod.EncodeBase64()
        assert enc_op.infer_shape() == ()
        assert enc_op.infer_shape(DummyWithShape((2, 3)), DummyWithShape((1, 3))) == (2, 3)

        dec_op = b64_mod.DecodeBase64()
        assert dec_op.infer_shape() == ()
        assert dec_op.infer_shape(DummyWithShape((4, 1)), DummyWithShape((1, 5))) == (4, 5)
    finally:
        config.eager_mode = orig_eager
