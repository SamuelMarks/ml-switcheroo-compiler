# ruff: noqa: E501
"""Tensorflow Code Generator Package."""

from .eager import execute_op
from .generator import TensorFlowCodeGenerator

TensorFlowCodeGenerator.execute_op = classmethod(execute_op)
