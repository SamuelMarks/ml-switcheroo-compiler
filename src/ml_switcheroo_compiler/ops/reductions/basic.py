"""Defines reduction operations for tensor computations.

This module contains the base class for reduction operations and concrete
implementations of common reductions such as Sum, Mean, Max, Min, Prod, Variance, Std,
Argmax, Argmin, All, Logsumexp, CountNonzero, Norm, Cumsum, and Any
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class ReductionOp(OpDef):
    """Base class for reduction operations.

    Provides common functionality for operations that reduce one or more dimensions
    of an input tensor, such as shape inference, NumPy evaluation, and argument
    formatting
    """

    op_name: str = ""

    def infer_shape(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return ()  # Symbolic shape inference will handle axis reduction logic

    def numpy_eval(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return getattr(np, getattr(self, "np_op_name", self.op_name.lower()))(
            x,
            axis=axis,
            keepdims=keepdims,
            **kwargs,
        )

    def _format_args(self, x: str, **kwargs: object) -> str:
        """Format args.

        Args:
            x (str): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            str: The resulting output
        """
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if kwargs.get("keepdims"):
            args.append(f"keepdims={kwargs['keepdims']}")
        return ", ".join(args)


@register_op("Sum")
class Sum(ReductionOp):
    """Sum reduction operation.

    Computes the sum of elements across specified dimensions of an input tensor
    """

    op_name = "Sum"


@register_op("Mean")
class Mean(ReductionOp):
    """Mean reduction operation.

    Computes the arithmetic mean of elements across specified dimensions of an input
    tensor
    """

    op_name = "Mean"


@register_op("Max")
class Max(ReductionOp):
    """Max reduction operation.

    Computes the maximum value of elements across specified dimensions of an input
    tensor
    """

    op_name = "Max"

    def numpy_eval(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.max(x, axis=axis, keepdims=keepdims, **kwargs)


@register_op("Min")
class Min(ReductionOp):
    """Min reduction operation.

    Computes the minimum value of elements across specified dimensions of an input
    tensor
    """

    op_name = "Min"

    def numpy_eval(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.min(x, axis=axis, keepdims=keepdims, **kwargs)


@register_op("Prod")
class Prod(ReductionOp):
    """Product reduction operation.

    Computes the product of elements across specified dimensions of an input tensor
    """

    op_name = "Prod"
    np_op_name = "prod"


@register_op("Variance")
class Variance(ReductionOp):
    """Variance reduction operation.

    Computes the variance of elements across specified dimensions of an input
    tensor
    """

    op_name = "Variance"
    np_op_name = "var"


@register_op("Std")
class Std(ReductionOp):
    """Standard deviation reduction operation.

    Computes the standard deviation of elements across specified dimensions of an
    input tensor
    """

    op_name = "Std"
    np_op_name = "std"


@register_op("Argmax")
class Argmax(ReductionOp):
    """Argmax reduction operation.

    Computes the indices of the maximum values across specified dimensions of an
    input tensor
    """

    op_name = "Argmax"
    np_op_name = "argmax"


@register_op("Argmin")
class Argmin(ReductionOp):
    """Argmin reduction operation.

    Computes the indices of the minimum values across specified dimensions of an
    input tensor
    """

    op_name = "Argmin"
    np_op_name = "argmin"


@register_op("All")
class All(ReductionOp):
    """Logical AND reduction operation.

    Checks if all elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "All"
    np_op_name = "all"


@register_op("Logsumexp")
class Logsumexp(ReductionOp):
    """Log-sum-exp reduction operation.

    Computes the logarithm of the sum of exponentials of elements across specified
    dimensions
    """

    op_name = "Logsumexp"
    np_op_name = "logsumexp"

    def numpy_eval(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import numpy as np

        xmax = np.max(x, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=keepdims)) + (
            np.squeeze(xmax, axis=axis) if not keepdims else xmax
        )


@register_op("CountNonzero")
class CountNonzero(ReductionOp):
    """Count non-zero elements reduction operation.

    Counts the number of non-zero elements across specified dimensions of an input
    tensor
    """

    op_name = "CountNonzero"
    np_op_name = "count_nonzero"


@register_op("Norm")
class Norm(ReductionOp):
    """Matrix or vector norm reduction operation.

    Computes the norm of elements across specified dimensions of an input tensor
    """

    op_name = "Norm"
    np_op_name = "norm"

    def numpy_eval(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            keepdims (bool): The keepdims parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import numpy as np

        return np.linalg.norm(x, axis=axis, keepdims=keepdims, **kwargs)


@register_op("Cumsum")
class Cumsum(ReductionOp):
    """Cumulative sum reduction operation.

    Computes the cumulative sum of elements across specified dimensions of an input
    tensor
    """

    op_name = "Cumsum"
    np_op_name = "cumsum"

    def numpy_eval(self, x: object, axis: object = None, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axis (object): The axis parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        # cumsum does not take keepdims
        import numpy as np

        return np.cumsum(x, axis=axis, **kwargs)


@register_op("Any")
class AnyOp(ReductionOp):
    """Logical OR reduction operation.

    Checks if any elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "Any"
    np_op_name = "any"
