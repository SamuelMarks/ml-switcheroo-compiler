# ruff: noqa: E402, E501, F403
"""Module __init__.py."""

"""Numpy Code Generator Package."""

from .eager import execute_op
from .generator import NumpyGenerator

__all__ = ["NumpyGenerator", "execute_op"]
