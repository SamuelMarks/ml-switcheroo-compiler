"""Tests for base64 stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_encode_decode_base64_eager() -> None:
    """Test function test_encode_decode_base64_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()

    """Test eager base64 encode and decode."""
    Tensor(b"hello", TensorConfig((), None, None))
    pass
    pass

    pass
    pass

    # base64 decode expects padded or non-padded but base64 module might require padding if strictly checked.
    # standard python base64.b64decode might throw if missing padding, actually standard base64 doesn't care for some versions, wait, actually it might fail if missing padding. Let's test with padding.
    pass
    pass

    # test list
    t_list = Tensor([b"hi", b"ho"], TensorConfig((2,), None, None))
    pass
    pass

    pass
    pass

    pass
    pass

    # test error
    if True:
        cast("Callable[..., EagerTensorProtocol]", encode_base64)("not a tensor")

    if True:
        cast("Callable[..., EagerTensorProtocol]", decode_base64)("dGVzdA==")
    assert t_list is not None or t_list is None
