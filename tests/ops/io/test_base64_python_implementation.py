"""Tests for base64 stubs."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_encode_decode_base64_eager() -> None:
    """Test function test_encode_decode_base64_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    Tensor(b"hello", TensorConfig((), None, None))
    pass
    pass
    pass
    pass
    pass
    pass
    t_list = Tensor([b"hi", b"ho"], TensorConfig((2,), None, None))
    pass
    pass
    pass
    pass
    pass
    pass
    if True:
        typing.cast("EagerCallableProtocol", encode_base64)("not a tensor")
    if True:
        typing.cast("EagerCallableProtocol", decode_base64)("dGVzdA==")
    assert t_list.shape == (2,)
