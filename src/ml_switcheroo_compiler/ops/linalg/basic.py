"""Defines linear algebra operations for the ML Switcheroo framework.

This module contains operator definitions (OpDefs) for common linear algebra
computations such as matrix multiplication, dot products, and Einstein summation,
supporting both shape inference and NumPy-based evaluation
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Matmul")
class Matmul(OpDef):
    """Matrix multiplication operator.

    Computes the matrix product of two arrays
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The a parameter
            b (object): The b parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) >= 2 and len(b) >= 2:
                # Basic matmul shape inference for 2D+
                return a[:-1] + b[1:]
        return None

    def numpy_eval(self, a: object, b: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            a (object): The a parameter
            b (object): The b parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.matmul(a, b)


@register_op("Dot")
class Dot(OpDef):
    """Dot product operator.

    Computes the dot product of two arrays
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The a parameter
            b (object): The b parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return None

    def numpy_eval(self, a: object, b: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            a (object): The a parameter
            b (object): The b parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.dot(a, b)


@register_op("Einsum")
class Einsum(OpDef):
    """Einstein summation operator.

    Evaluates the Einstein summation convention on the operands
    """

    def infer_shape(
        self,
        subscripts: str,
        *operands: object,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            subscripts (str): The subscripts parameter
            *operands (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return None

    def numpy_eval(
        self,
        subscripts: str,
        *operands: object,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            subscripts (str): The subscripts parameter
            *operands (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.einsum(subscripts, *operands)
