# ruff: noqa: E501
"""Mlx Code Generator Package."""

from .eager import execute_op
from .generator import MLXCodeGenerator

MLXCodeGenerator.execute_op = classmethod(execute_op)
