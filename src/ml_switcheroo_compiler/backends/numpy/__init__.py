"""Numpy Code Generator Package."""

from .eager import execute_op
from .generator import NumpyGenerator

__all__ = ["NumpyGenerator", "execute_op"]
