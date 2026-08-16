# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

from typing import Any

"""Pytorch Code Generator Package."""

from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

PyTorchCodeGenerator.zeros = classmethod(zeros)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
PyTorchCodeGenerator.array = classmethod(array)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
PyTorchCodeGenerator.asarray = classmethod(asarray)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
PyTorchCodeGenerator.item = classmethod(item)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
PyTorchCodeGenerator.execute_op = classmethod(execute_op)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
