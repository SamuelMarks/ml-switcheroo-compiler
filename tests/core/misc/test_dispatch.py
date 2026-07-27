"""Test module."""

from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dispatch import dispatch


def test_dispatch():
    class DummySub:
        def func(self, *a, **k):
            return "res"

    class DummyMod:
        mod = DummySub()

    class DummyBackend:
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
        dtype = "float32"
