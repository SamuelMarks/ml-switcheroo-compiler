"""Tests for test_mlx_distributed."""

from __future__ import annotations

import typing

import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_mlx_all_to_all_fallback(mocker: MockerFixture) -> None:
    """Test all to all fallback.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_to_all

    class DummyCore(MockTestStub):
        """Test class DummyCore."""

        pass

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    assert _mlx_all_to_all(None, "tensor") == "tensor"


def test_mlx_reduce_scatter_exception(mocker: MockerFixture) -> None:
    """Test reduce scatter exception.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter

    class DummyGroup(MockTestStub):
        """Test class DummyGroup."""

        def rank(self) -> None:
            """Test function rank.

            Raises:
                Exception: When raised during `rank` (No).
            """
            raise RuntimeError("No")

    class DummyDistributed:
        """Test class DummyDistributed."""

        def all_sum(self, tensor: BackendArg) -> BackendArg:
            """Test function all_sum.

            Args:
                tensor: Test parameter `tensor`.

            Returns:
                BackendArg: Computed output value from `all_sum`.
            """

            class DummyTensor:
                """Test class DummyTensor."""

                shape = [10]
                ndim = 1

                def __getitem__(self, idx: BackendArg) -> BackendArg:
                    """Test function __getitem__.

                    Args:
                        idx: Test parameter `idx`.

                    Returns:
                        BackendArg: Computed output value from `__getitem__`.
                    """
                    return "sliced"

            return DummyTensor()

    class DummyCore(MockTestStub):
        """Test class DummyCore."""

        distributed = DummyDistributed()

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    res = _mlx_reduce_scatter(DummyMLX.core, "tensor", group=DummyGroup())
    assert res == "sliced"


def test_mlx_reduce_scatter_not_supported(mocker: MockerFixture) -> None:
    """Test reduce scatter not supported.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    class DummyCore(MockTestStub):
        """Test class DummyCore."""

        pass

    class DummyMLX:
        """Test class DummyMLX."""

        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    with pytest.raises(BackendNotSupportedError, match="not supported"):
        _mlx_reduce_scatter(None, "tensor")
