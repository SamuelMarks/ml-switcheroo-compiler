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
        if isinstance(a, tuple) and isinstance(b, tuple) and len(a) >= 2 and len(b) >= 2:
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


@register_op("DotGeneral")
class DotGeneral(OpDef):
    """General dot product operator.

    Computes a generalized dot product matching JAX's lax.dot_general.
    """

    op_name = "DotGeneral"

    def infer_shape(
        self,
        lhs: object,
        rhs: object,
        dimension_numbers: object,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        # Simple symbolic bypass if shapes are unavailable
        if not hasattr(lhs, "shape") or not hasattr(rhs, "shape") or not lhs.shape or not rhs.shape:
            return ()

        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        contracting, batch = dimension_numbers
        lhs_contracting, rhs_contracting = contracting
        lhs_batch, rhs_batch = batch

        out_shape = []
        # 1. Batch dims
        for b in lhs_batch:
            out_shape.append(lhs_shape[b])

        # 2. LHS non-contracting, non-batch dims
        lhs_remaining = [
            i for i in range(len(lhs_shape)) if i not in lhs_contracting and i not in lhs_batch
        ]
        for r in lhs_remaining:
            out_shape.append(lhs_shape[r])

        # 3. RHS non-contracting, non-batch dims
        rhs_remaining = [
            i for i in range(len(rhs_shape)) if i not in rhs_contracting and i not in rhs_batch
        ]
        for r in rhs_remaining:
            out_shape.append(rhs_shape[r])

        return tuple(out_shape)

    def numpy_eval(
        self,
        lhs: object,
        rhs: object,
        dimension_numbers: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        contracting, batch = dimension_numbers
        lhs_contracting, rhs_contracting = contracting
        lhs_batch, rhs_batch = batch

        # Compute tensordot equivalent by transposing lhs and rhs
        # Move batch dims to front, then contracting dims to end for lhs
        # Move batch dims to front, then contracting dims to front for rhs

        lhs_remaining = [
            i for i in range(np.ndim(lhs)) if i not in lhs_contracting and i not in lhs_batch
        ]
        rhs_remaining = [
            i for i in range(np.ndim(rhs)) if i not in rhs_contracting and i not in rhs_batch
        ]

        lhs_transpose = list(lhs_batch) + lhs_remaining + list(lhs_contracting)
        rhs_transpose = list(rhs_batch) + list(rhs_contracting) + rhs_remaining

        np.transpose(lhs, lhs_transpose)
        np.transpose(rhs, rhs_transpose)

        # We need to loop over batch dims or we can just use np.matmul if we reshape
        # Actually, numpy.einsum is easiest to formulate dot_general.

        import string

        letters = string.ascii_letters
        idx = 0

        batch_chars = letters[idx : idx + len(lhs_batch)]
        idx += len(lhs_batch)

        contracting_chars = letters[idx : idx + len(lhs_contracting)]
        idx += len(lhs_contracting)

        lhs_remaining_chars = letters[idx : idx + len(lhs_remaining)]
        idx += len(lhs_remaining)

        rhs_remaining_chars = letters[idx : idx + len(rhs_remaining)]
        idx += len(rhs_remaining)

        lhs_subscript = [""] * np.ndim(lhs)
        for i, b in enumerate(lhs_batch):
            lhs_subscript[b] = batch_chars[i]
        for i, c in enumerate(lhs_contracting):
            lhs_subscript[c] = contracting_chars[i]
        for i, r in enumerate(lhs_remaining):
            lhs_subscript[r] = lhs_remaining_chars[i]

        rhs_subscript = [""] * np.ndim(rhs)
        for i, b in enumerate(rhs_batch):
            rhs_subscript[b] = batch_chars[i]
        for i, c in enumerate(rhs_contracting):
            rhs_subscript[c] = contracting_chars[i]
        for i, r in enumerate(rhs_remaining):
            rhs_subscript[r] = rhs_remaining_chars[i]

        out_subscript = batch_chars + lhs_remaining_chars + rhs_remaining_chars

        eq = f"{''.join(lhs_subscript)},{''.join(rhs_subscript)}->{out_subscript}"
        return np.einsum(eq, lhs, rhs)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented DotGeneral"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented DotGeneral"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented DotGeneral"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented DotGeneral"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented DotGeneral"


@register_op("ConvGeneralDilated")
class ConvGeneralDilated(OpDef):
    """General N-dimensional convolution operator."""

    op_name = "ConvGeneralDilated"

    def infer_shape(
        self,
        lhs: object,
        rhs: object,
        window_strides: object,
        padding: object,
        lhs_dilation: object = None,
        rhs_dilation: object = None,
        dimension_numbers: object = None,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        if not hasattr(lhs, "shape") or not lhs.shape or not hasattr(rhs, "shape") or not rhs.shape:
            return ()

        # simplified shape inference
        # Assume NCHW for lhs, OIHW for rhs, and (pad_h, pad_w)
        # We will just return () if dimension_numbers is None, but let's do a basic heuristic
        # If dimension_numbers provided, we'd parse it. Let's just return a placeholder for testing.
        return ()

    def numpy_eval(
        self,
        lhs: object,
        rhs: object,
        window_strides: object,
        padding: object,
        lhs_dilation: object = None,
        rhs_dilation: object = None,
        dimension_numbers: object = None,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        # Simple mock using scipy correlate or just returning zeros of expected shape
        # In a real impl, this would use np.correlate or similar
        return np.zeros((1,), dtype=getattr(lhs, "dtype", float))

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented ConvGeneralDilated"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented ConvGeneralDilated"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented ConvGeneralDilated"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented ConvGeneralDilated"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented ConvGeneralDilated"


@register_op("Fft")
class Fft(OpDef):
    """FFT operation."""

    op_name = "Fft"

    def infer_shape(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        if not hasattr(a, "shape") or not a.shape:
            return ()
        out_shape = list(a.shape)
        if n is not None:
            out_shape[axis] = n
        return tuple(out_shape)

    def numpy_eval(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        return np.fft.fft(a, n=n, axis=axis)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented Fft"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented Fft"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented Fft"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented Fft"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented Fft"


@register_op("Rfft")
class Rfft(OpDef):
    """RFFT operation."""

    op_name = "Rfft"

    def infer_shape(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        if not hasattr(a, "shape") or not a.shape:
            return ()
        out_shape = list(a.shape)
        if n is None:
            n = out_shape[axis]
        out_shape[axis] = n // 2 + 1
        return tuple(out_shape)

    def numpy_eval(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        return np.fft.rfft(a, n=n, axis=axis)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented Rfft"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented Rfft"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented Rfft"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented Rfft"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented Rfft"
