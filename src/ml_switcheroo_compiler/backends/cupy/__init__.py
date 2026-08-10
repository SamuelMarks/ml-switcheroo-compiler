# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Cupy Code Generator Package."""

from .eager import execute_op
from .generator import CupyGenerator
from .types import array, asarray, item, zeros

CupyGenerator.zeros = classmethod(zeros)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
CupyGenerator.array = classmethod(array)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
CupyGenerator.asarray = classmethod(asarray)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
CupyGenerator.item = classmethod(item)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
CupyGenerator.execute_op = classmethod(execute_op)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
