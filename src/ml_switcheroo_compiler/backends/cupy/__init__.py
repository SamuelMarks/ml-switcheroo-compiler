# ruff: noqa: E501
"""Cupy Code Generator Package."""

from .eager import execute_op
from .generator import CupyGenerator
from .types import array, asarray, item, zeros

CupyGenerator.zeros = classmethod(zeros)
CupyGenerator.array = classmethod(array)
CupyGenerator.asarray = classmethod(asarray)
CupyGenerator.item = classmethod(item)
CupyGenerator.execute_op = classmethod(execute_op)
