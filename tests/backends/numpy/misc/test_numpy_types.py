"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_numpy_types() -> None:
    """Test function test_numpy_types."""
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", zeros)(None, (1,)), "shape")) == (1,)

    assert typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", array)(None, [1]), "tolist")))() == [1]
    import numpy as np

    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", array)(None, [1], dtype=np.int32), "dtype")) == np.int32

    assert typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", asarray)(None, [1]), "tolist")))() == [1]

    assert typing.cast("EagerCallableProtocol", item)(None, [42]) == 42.0
