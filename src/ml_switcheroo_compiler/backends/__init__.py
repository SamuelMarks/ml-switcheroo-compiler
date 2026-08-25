# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

"""Backend code generators for ML Switcheroo Compiler."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend

__all__ = [
    "BackendRegistry",
    "BaseGenerator",
    "register_backend",
]

__all__ = [
    "BackendRegistry",
    "BaseGenerator",
    "register_backend",
]
