"""Tests for test_config_eager."""

from __future__ import annotations

import os

from ml_switcheroo_compiler.core.config import config


def test_config_eager_mode_env_override() -> None:
    """Test eager_mode property when SWITCHEROO_EAGER_MODE environment variable is set to 1."""
    orig_env: str | None = os.environ.get("SWITCHEROO_EAGER_MODE")
    orig_mode: bool = config.eager_mode
    try:
        config.eager_mode = False
        os.environ["SWITCHEROO_EAGER_MODE"] = "1"
        assert config.eager_mode is True
    finally:
        if orig_env is None:
            os.environ.pop("SWITCHEROO_EAGER_MODE", None)
        else:
            os.environ["SWITCHEROO_EAGER_MODE"] = orig_env
        config.eager_mode = orig_mode
