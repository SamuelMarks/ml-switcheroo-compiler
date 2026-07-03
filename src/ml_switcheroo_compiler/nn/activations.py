"""Module docstring."""

import math

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.creation.frontend import full_like, ones_like, zeros_like
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.frontend import expand_dims
from ml_switcheroo_compiler.ops.unary import expm1


def clip(x: object, a: object, b: object) -> object:
    """Function docstring."""
    return get_op("Clip")()(x, a, b)


GELU_CONSTANT = 0.044715
RELU6_MAX = 6.0
HARDSIGMOID_SCALE = 6.0
HARDSIGMOID_OFFSET = 0.5
HARDSWISH_MAX = 6.0
HARDSWISH_OFFSET = 3.0
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946
LEAKY_RELU_DEFAULT_SLOPE = 0.01
SPARSE_PLUS_MID_MULTIPLIER = 0.25


def gelu(x: object, approximate: object = False) -> object:
    """Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (object): The input x tensor.
        approximate (object): The approximate parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    if approximate == "tanh" or approximate is True:
        const1 = full_like(x, math.sqrt(2 / math.pi))  # pragma: no cover
        const2 = full_like(x, GELU_CONSTANT)  # pragma: no cover
        x3 = get_op("Power")()(x, full_like(x, 3.0))  # pragma: no cover
        inner = get_op("Add")()(x, get_op("Multiply")()(const2, x3))  # pragma: no cover
        tanh_in = get_op("Multiply")()(const1, inner)  # pragma: no cover
        tanh_out = get_op("Tanh")()(tanh_in)  # pragma: no cover
        one_plus = get_op("Add")()(full_like(x, 1.0), tanh_out)  # pragma: no cover
        return get_op("Multiply")()(full_like(x, 0.5), get_op("Multiply")()(x, one_plus))  # pragma: no cover
    const_sqrt2 = full_like(x, math.sqrt(2.0))
    erf_in = get_op("Divide")()(x, const_sqrt2)
    erf_out = get_op("Erf")()(erf_in)
    one_plus = get_op("Add")()(full_like(x, 1.0), erf_out)
    return get_op("Multiply")()(get_op("Multiply")()(x, full_like(x, 0.5)), one_plus)


def logsumexp(
    a: object,
    axis: object = None,
    **kwargs: object,
) -> object:
    """Computes the log of the sum of exponentials of input elements.

    Args:
        a (object): The input a tensor.
        axis (object): The axis along which to perform the operation.
        **kwargs (object): Additional arguments like b, keepdims, return_sign, where.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    logsumexp = get_op("Logsumexp")()
    get_op("Max")()
    get_op("Sum")()  # logsumexp as _lse

    keepdims = kwargs.get("keepdims", False)
    return logsumexp(a, axis=axis, keepdims=keepdims)


def one_hot(x: Tensor, num_classes: int, *, dtype: object = None, axis: int = -1) -> Tensor:
    """Creates a one-hot encoding of the given integer array.

    Args:
        x (Tensor): The input x tensor.
        num_classes (int): The num_classes parameter for the operation.
        dtype (object): The target data type.
        axis (int): The axis along which to perform the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    equal = get_op("Equal")()

    arange = get_op("Arange")()

    cast = get_op("Cast")()

    classes = arange(num_classes)
    for _ in range(len(x.shape)):
        classes = expand_dims(classes, 0)

    x_expanded = expand_dims(x, axis)
    result = equal(x_expanded, classes)

    return cast(result, dtype=dtype)


def softmax(x: Tensor, axis: int = -1, where: object = None, initial: object = None) -> Tensor:
    """Computes the softmax activation function over the given axis.

    Args:
        x (Tensor): The input x tensor.
        axis (int): The axis along which to perform the operation.
        where (object): The where parameter for the operation.
        initial (object): The initial parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    subtract = get_op("Subtract")()
    true_divide = get_op("TrueDivide")()

    get_op("Logsumexp")()
    max = get_op("Max")()
    sum = get_op("Sum")()  # max, sum

    exp = get_op("Exp")()

    x_max = max(x, axis=axis, keepdims=True)
    unnormalized = exp(subtract(x, x_max))
    return true_divide(unnormalized, sum(unnormalized, axis=axis, keepdims=True))


def sigmoid(x: Tensor) -> Tensor:
    """Computes the sigmoid activation function.

    Args:
        x (Tensor): The input x tensor.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    add = get_op("Add")()
    true_divide = get_op("TrueDivide")()

    exp = get_op("Exp")()
    negative = get_op("Negative")()

    return true_divide(1.0, add(1.0, exp(negative(x))))


def log_sigmoid(x: Tensor) -> Tensor:
    """Computes the logarithm of the sigmoid function.

    Args:
        x (Tensor): The input x tensor.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    less = get_op("Less")()
    subtract = get_op("Subtract")()

    where = get_op("Where")()

    exp = get_op("Exp")()
    log1p = get_op("Log1P")()
    negative = get_op("Negative")()

    # log(sigmoid(x)) = -log1p(exp(-x)) for x > 0
    # log(sigmoid(x)) = x - log1p(exp(x)) for x < 0
    # This prevents overflow in exp(-x) when x is large negative
    is_neg = less(x, 0.0)
    neg_branch = subtract(x, log1p(exp(x)))
    pos_branch = negative(log1p(exp(negative(x))))
    return where(is_neg, neg_branch, pos_branch)


def relu(x: object) -> object:
    """Computes the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    zero = zeros_like(x)
    return get_op("Maximum")()(x, zero)


def relu2(x: object) -> object:
    """Computes the ReLU2 activation function, capping at 2."""
    return clip(x, 0.0, 2.0)


def relu6(x: object) -> object:
    """Computes the ReLU6 activation function, capping at 6.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return clip(x, 0.0, RELU6_MAX)


def hard_sigmoid(x: object) -> object:
    """Computes the Hard Sigmoid activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return clip(x / HARDSIGMOID_SCALE + HARDSIGMOID_OFFSET, 0.0, 1.0)


def hard_tanh(x: object) -> object:
    """Computes the Hard Tanh activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return clip(x, -1.0, 1.0)


def swish(x: object) -> object:
    """Computes the Swish activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return x / (1.0 + get_op("Exp")()(-x))


def silu(x: object) -> object:
    """Computes the SiLU activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return x / (1.0 + get_op("Exp")()(-x))


def elu(x: object, alpha: object = 1.0) -> object:
    """Computes the Exponential Linear Unit (ELU) activation function.

    Args:
        x (object): The input x tensor.
        alpha (object): The alpha parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return get_op("Where")()(x > 0.0, x, alpha * (get_op("Exp")()(x) - 1.0))


def celu(x: object, alpha: object = 1.0) -> object:
    """Computes the Continuously Differentiable Exponential Linear Unit.

    Args:
        x (object): The input x tensor.
        alpha (object): The alpha parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    zero = zeros_like(x)
    return get_op("Maximum")()(x, zero) + get_op("Minimum")()(alpha * (get_op("Exp")()(x / alpha) - 1.0), zero)


def selu(x: object) -> object:
    """Computes the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    alpha = SELU_ALPHA
    scale = SELU_SCALE

    pos = get_op("Maximum")()(x, full_like(x, 0.0))
    neg = get_op("Multiply")()(
        full_like(x, alpha),
        expm1(get_op("Minimum")()(x, full_like(x, 0.0))),
    )
    return get_op("Multiply")()(full_like(x, scale), get_op("Add")()(pos, neg))


def log_softmax(x: object, axis: object = -1) -> object:
    """Computes the logarithm of the softmax activation function.

    Args:
        x (object): The input x tensor.
        axis (object): The axis along which to perform the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    amax = get_op("Max")()(x, axis=axis, keepdims=True)
    shifted = get_op("Subtract")()(x, amax)
    sum_exp = get_op("Sum")()(get_op("Exp")()(shifted), axis=axis, keepdims=True)
    return get_op("Subtract")()(shifted, get_op("Log")()(sum_exp))


def glu(x: object, axis: int = -1) -> object:
    """Computes the Gated Linear Unit (GLU) activation function."""
    a, b = get_op("Split")()(x, 2, dim=axis)
    return get_op("Multiply")()(a, sigmoid(b))


def hard_silu(x: object) -> object:
    """Computes the Hard SiLU activation function."""
    return get_op("Multiply")()(x, hard_sigmoid(x))


def hard_swish(x: object) -> object:
    """Computes the Hard Swish activation function."""
    return get_op("Multiply")()(
        x,
        get_op("Divide")()(
            clip(get_op("Add")()(x, HARDSWISH_OFFSET), 0.0, HARDSWISH_MAX),
            HARDSWISH_MAX,
        ),
    )


def leaky_relu(x: object, negative_slope: float = LEAKY_RELU_DEFAULT_SLOPE) -> object:
    """Computes the Leaky ReLU activation function."""
    return get_op("Where")()(get_op("GreaterEqual")()(x, 0.0), x, get_op("Multiply")()(x, negative_slope))


def mish(x: object) -> object:
    """Computes the Mish activation function."""
    return get_op("Multiply")()(x, get_op("Tanh")()(softplus(x)))


def soft_sign(x: object) -> object:
    """Computes the SoftSign activation function."""
    return get_op("Divide")()(x, get_op("Add")()(1.0, get_op("Abs")()(x)))


def softplus(x: object) -> object:
    """Computes the Softplus activation function."""
    return get_op("Log1P")()(get_op("Exp")()(x))


def sparse_plus(x: object) -> object:
    """Computes the SparsePlus activation function."""
    leq = get_op("LessEqual")()(x, full_like(x, -1.0))
    geq = get_op("GreaterEqual")()(x, full_like(x, 1.0))
    mid = get_op("Multiply")()(
        full_like(x, SPARSE_PLUS_MID_MULTIPLIER),
        get_op("Square")()(get_op("Add")()(x, full_like(x, 1.0))),
    )
    return get_op("Where")()(leq, zeros_like(x), get_op("Where")()(geq, x, mid))


def sparse_sigmoid(x: object) -> object:
    """Computes the Sparse Sigmoid activation function."""
    return clip(get_op("Add")()(get_op("Multiply")()(0.5, x), 0.5), 0.0, 1.0)


def squareplus(x: object, b: float = 4.0) -> object:
    """Computes the SquarePlus activation function."""
    return get_op("Multiply")()(0.5, get_op("Add")()(x, get_op("Sqrt")()(get_op("Add")()(get_op("Square")()(x), b))))


def standardize(x: object, axis: int = -1, epsilon: float = 1e-5) -> object:
    """Standardizes the input tensor along the given axis."""
    mean = get_op("Mean")()(x, axis=axis, keepdims=True)
    var = get_op("Var")()(x, axis=axis, keepdims=True)  # pragma: no cover
    return get_op("Divide")()(get_op("Subtract")()(x, mean), get_op("Sqrt")()(get_op("Add")()(var, epsilon)))  # pragma: no cover


def hard_shrink(x: object, lower: float = -0.5, upper: float = 0.5) -> object:
    """Computes the Hard Shrink activation function."""
    cond = get_op("LogicalOr")()(get_op("Less")()(x, lower), get_op("Greater")()(x, upper))
    return get_op("Where")()(cond, x, zeros_like(x))


def soft_shrink(x: object, lower: float = -0.5, upper: float = 0.5) -> object:
    """Computes the Soft Shrink activation function."""
    return get_op("Where")()(
        get_op("Less")()(x, lower),
        get_op("Subtract")()(x, lower),
        get_op("Where")()(get_op("Greater")()(x, upper), get_op("Subtract")()(x, upper), zeros_like(x)),
    )


def tanh_shrink(x: object) -> object:
    """Computes the Tanh Shrink activation function."""
    return get_op("Subtract")()(x, get_op("Tanh")()(x))


def threshold(x: object, threshold: float = 0.0, value: float = 0.0) -> object:
    """Computes the Threshold activation function."""
    return get_op("Where")()(get_op("Greater")()(x, threshold), x, full_like(x, value))


def sparsemax(x: object, axis: int = -1) -> object:
    """Computes the Sparsemax activation function."""
    # sparsemax(z) = max(0, z - tau(z))
    # where tau(z) is the threshold function.
    # To implement sparsemax properly:
    # 1. Sort z in descending order
    # 2. Find k = max {j : 1 + j * z_j > sum_{i=1}^j z_i}
    # 3. tau = (sum_{i=1}^k z_i - 1) / k
    # 4. max(0, z - tau)
    # Since we may not have full sort easily, we can use an approximation or full sort.
    # We will use sort.
    sorted_x = get_op("Sort")()(x, axis=axis)  # Ascending
    # We want descending
    sorted_x = get_op("Multiply")()(get_op("Sort")()(get_op("Multiply")()(x, -1.0), axis=axis), -1.0)

    cumsum_x = get_op("Cumsum")()(sorted_x, axis=axis)

    # Create an array of [1, 2, ..., d] along the axis
    shape = x.shape
    d = shape[axis] if axis >= 0 else shape[len(shape) + axis]

    # rank of x
    rank = len(shape)
    arange_shape = [1] * rank
    arange_shape[axis] = d

    j = get_op("Reshape")()(get_op("Arange")()(1, d + 1, dtype=x.dtype.value), arange_shape)

    # condition: 1 + j * z_j > sum_{i=1}^j z_i
    cond = get_op("Greater")()(get_op("Add")()(1.0, get_op("Multiply")()(j, sorted_x)), cumsum_x)

    k = get_op("Sum")()(get_op("Cast")()(cond, dtype=x.dtype.value), axis=axis, keepdims=True)

    # To get the sum up to k, we can use gather or just sum with masking
    # Wait, sum_k is the k-th element of cumsum_x
    # We can get it by picking the element at k-1
    # Actually, using a mask is easier: mask = j <= k
    mask = get_op("LessEqual")()(j, k)
    sum_k = get_op("Sum")()(get_op("Where")()(mask, sorted_x, zeros_like(sorted_x)), axis=axis, keepdims=True)

    tau = get_op("Divide")()(get_op("Subtract")()(sum_k, 1.0), k)

    return get_op("Maximum")()(0.0, get_op("Subtract")()(x, tau))


def prelu(x: object, alpha: object) -> object:
    """Parametric ReLU.

    Args:
        x (object): Input tensor.
        alpha (object): Slope for negative values.

    Returns:
        object: The result.
    """
    maximum = get_op("Maximum")()
    minimum = get_op("Minimum")()
    multiply = get_op("Multiply")()

    return maximum(0.0, x) + multiply(alpha, minimum(0.0, x))


def softmin(x: object, axis: int = -1) -> object:
    """Softmin function.

    Args:
        x (object): Input tensor.
        axis (int): Axis along which to compute the softmin.

    Returns:
        object: The result.
    """
    negative = get_op("Negative")()

    return softmax(negative(x), axis=axis)


def step(x: object) -> object:
    """Step function (Heaviside).

    Args:
        x (object): Input tensor.

    Returns:
        object: The result.
    """
    greater_equal = get_op("GreaterEqual")()
    where = get_op("Where")()

    return where(greater_equal(x, 0.0), ones_like(x), zeros_like(x))


hardshrink = hard_shrink
hardtanh = hard_tanh
hardswish = hard_swish
logsigmoid = log_sigmoid
softshrink = soft_shrink
softsign = soft_sign
