"""Module weibull_min.py."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for weibull_min.py."""

from ml_switcheroo_compiler.random.state import _dispatch_random


def weibull_min(*args: Any, **kwargs: Any) -> Tensor:
    """Evaluate weibull_min operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    return _dispatch_random("weibull_min", *args, **kwargs)
