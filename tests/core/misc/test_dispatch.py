"""Test module."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dispatch import dispatch

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_dispatch() -> None:
    """Test function test_dispatch."""

    class DummySub:
        """Test class DummySub."""

        def func(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function func.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "res"

    class DummyMod:
        """Test class DummyMod."""

        mod = DummySub()

    class DummyBackend:
        """Test class DummyBackend."""

        module = DummyMod()

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.core.dispatch.get_active_backend", return_value=DummyBackend()):
        assert dispatch("mod", "func") == "res"

        with pytest.raises(ValueError):
            dispatch("not_mod", "func")

        with pytest.raises(ValueError):
            dispatch("mod", "not_func")

    config.eager_mode = False

    class DummyTensor:
        """Test class DummyTensor."""

        dtype = "float32"
