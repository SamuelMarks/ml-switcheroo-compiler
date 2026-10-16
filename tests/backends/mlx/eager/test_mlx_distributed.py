"""Tests for test_mlx_distributed."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_all_to_all_fallback(mocker: EagerTensorProtocol) -> None:
    """Test all to all fallback."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_to_all

    class DummyCore:
        """Test class DummyCore."""

        pass

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    cast("Callable[..., EagerTensorProtocol]", mocker.patch.dict)(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    assert _mlx_all_to_all(None, "tensor") == "tensor"


def test_mlx_reduce_scatter_exception(mocker: EagerTensorProtocol) -> None:
    """Test reduce scatter exception."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter

    class DummyGroup:
        """Test class DummyGroup."""

        def rank(self) -> None:
            """Test function rank."""
            raise Exception("No")

    class DummyDistributed:
        """Test class DummyDistributed."""

        def all_sum(self, tensor: EagerValue) -> EagerValue:
            """Test function all_sum."""

            class DummyTensor:
                """Test class DummyTensor."""

                shape = [10]
                ndim = 1

                def __getitem__(self, idx: EagerValue) -> EagerValue:
                    """Test function __getitem__."""
                    return "sliced"

            return DummyTensor()

    class DummyCore:
        """Test class DummyCore."""

        distributed = DummyDistributed()

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    cast("Callable[..., EagerTensorProtocol]", mocker.patch.dict)(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    res = _mlx_reduce_scatter(DummyMLX.core, "tensor", group=DummyGroup())
    assert res == "sliced"


def test_mlx_reduce_scatter_not_supported(mocker: EagerTensorProtocol) -> None:
    """Test reduce scatter not supported."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    class DummyCore:
        """Test class DummyCore."""

        pass

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    cast("Callable[..., EagerTensorProtocol]", mocker.patch.dict)(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    with pytest.raises(BackendNotSupportedError, match="not supported"):
        _mlx_reduce_scatter(None, "tensor")
