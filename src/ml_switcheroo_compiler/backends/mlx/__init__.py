# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mlx Code Generator Package."""

import importlib.util

if importlib.util.find_spec("mlx") is None:
    raise ImportError("The 'mlx' backend requires the 'mlx' library to be installed.")

from .eager import execute_op
from .generator import MLXCodeGenerator

MLXCodeGenerator.execute_op = classmethod(execute_op)
