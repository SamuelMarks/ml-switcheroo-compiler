# ruff: noqa: E501
"""Jax Code Generator Package."""

from .eager import execute_op
from .generator import JAXCodeGenerator

JAXCodeGenerator.execute_op = classmethod(execute_op)
