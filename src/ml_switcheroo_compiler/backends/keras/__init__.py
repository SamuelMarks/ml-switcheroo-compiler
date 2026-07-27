# ruff: noqa: E501
"""Keras Code Generator Package."""

from .eager import execute_op
from .generator import KerasCodeGenerator

KerasCodeGenerator.execute_op = classmethod(execute_op)
