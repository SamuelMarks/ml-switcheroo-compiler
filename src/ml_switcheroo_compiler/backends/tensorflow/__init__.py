"""Tensorflow Code Generator Package."""

from .eager import execute_op
from .generator import TensorFlowCodeGenerator
from .types import array, asarray, item, zeros

TensorFlowCodeGenerator.zeros = classmethod(zeros)
TensorFlowCodeGenerator.array = classmethod(array)
TensorFlowCodeGenerator.asarray = classmethod(asarray)
TensorFlowCodeGenerator.item = classmethod(item)
TensorFlowCodeGenerator.execute_op = classmethod(execute_op)

__all__ = ["TensorFlowCodeGenerator"]
