"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_numpy_types() -> None:
    """Test function test_numpy_types."""
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", zeros)(None, (1,)), "shape")) == (1,)

    assert cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", array)(None, [1]), "tolist")))() == [1]
    import numpy as np

    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", array)(None, [1], dtype=np.int32), "dtype")) == np.int32

    assert cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", asarray)(None, [1]), "tolist")))() == [1]

    assert cast("EagerCallableProtocol", item)(None, [42]) == 42.0
