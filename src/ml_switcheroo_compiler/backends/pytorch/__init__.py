# ruff: noqa: E501
"""Pytorch Code Generator Package."""

from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

PyTorchCodeGenerator.zeros = classmethod(zeros)
PyTorchCodeGenerator.array = classmethod(array)
PyTorchCodeGenerator.asarray = classmethod(asarray)
PyTorchCodeGenerator.item = classmethod(item)
PyTorchCodeGenerator.execute_op = classmethod(execute_op)
