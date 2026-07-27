# ruff: noqa: E501
"""Numpy Code Generator Package."""

from .eager import execute_op
from .generator import NumpyGenerator

NumpyGenerator.execute_op = classmethod(execute_op)
