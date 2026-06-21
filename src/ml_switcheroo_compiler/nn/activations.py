"""Neural network modules and layers."""

from ml_switcheroo_compiler.core.tensor import Tensor


def gelu(x: object, approximate: object = False) -> object:
    """Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (object): The input x tensor.
        approximate (object): The approximate parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    import math

    from ml_switcheroo_compiler import ops

    if approximate == "tanh" or approximate is True:
        # 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        const1 = ops.full_like(x, math.sqrt(2 / math.pi))
        const2 = ops.full_like(x, 0.044715)
        x3 = ops.power(x, ops.full_like(x, 3.0))
        inner = ops.add(x, ops.multiply(const2, x3))
        tanh_in = ops.multiply(const1, inner)
        tanh_out = ops.tanh(tanh_in)
        one_plus = ops.add(ops.full_like(x, 1.0), tanh_out)
        return ops.multiply(ops.full_like(x, 0.5), ops.multiply(x, one_plus))
    # x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
    const_sqrt2 = ops.full_like(x, math.sqrt(2.0))
    erf_in = ops.divide(x, const_sqrt2)
    erf_out = ops.erf(erf_in)
    one_plus = ops.add(ops.full_like(x, 1.0), erf_out)
    return ops.multiply(ops.multiply(x, ops.full_like(x, 0.5)), one_plus)


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
    from ml_switcheroo_compiler.ops.reductions import logsumexp as _lse

    keepdims = kwargs.get("keepdims", False)
    return _lse(a, axis=axis, keepdims=keepdims)


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
    from ml_switcheroo_compiler.ops.binary import equal
    from ml_switcheroo_compiler.ops.creation.frontend import arange
    from ml_switcheroo_compiler.ops.shape.frontend import expand_dims
    from ml_switcheroo_compiler.ops.unary import cast

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
    from ml_switcheroo_compiler.ops.binary import subtract, true_divide
    from ml_switcheroo_compiler.ops.reductions import max, sum
    from ml_switcheroo_compiler.ops.unary import exp

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
    from ml_switcheroo_compiler.ops.binary import add, true_divide
    from ml_switcheroo_compiler.ops.unary import exp, negative

    return true_divide(1.0, add(1.0, exp(negative(x))))


def log_sigmoid(x: Tensor) -> Tensor:
    """Computes the logarithm of the sigmoid function.

    Args:
        x (Tensor): The input x tensor.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    from ml_switcheroo_compiler.ops.binary import less, subtract
    from ml_switcheroo_compiler.ops.shape.frontend import where
    from ml_switcheroo_compiler.ops.unary import exp, log1p, negative

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
    from ml_switcheroo_compiler import ops

    zero = ops.zeros_like(x)
    return ops.maximum(x, zero)


def relu6(x: object) -> object:
    """Computes the ReLU6 activation function, capping at 6.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return ops.clip(x, 0.0, 6.0)


def hard_sigmoid(x: object) -> object:
    """Computes the Hard Sigmoid activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return ops.clip(x / 6.0 + 0.5, 0.0, 1.0)


def hard_tanh(x: object) -> object:
    """Computes the Hard Tanh activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return ops.clip(x, -1.0, 1.0)


def swish(x: object) -> object:
    """Computes the Swish activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return x / (1.0 + ops.exp(-x))


def silu(x: object) -> object:
    """Computes the SiLU activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return x / (1.0 + ops.exp(-x))


def elu(x: object, alpha: object = 1.0) -> object:
    """Computes the Exponential Linear Unit (ELU) activation function.

    Args:
        x (object): The input x tensor.
        alpha (object): The alpha parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    return ops.where(x > 0.0, x, alpha * (ops.exp(x) - 1.0))


def celu(x: object, alpha: object = 1.0) -> object:
    """Computes the Continuously Differentiable Exponential Linear Unit.

    Args:
        x (object): The input x tensor.
        alpha (object): The alpha parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    zero = ops.zeros_like(x)
    return ops.maximum(x, zero) + ops.minimum(alpha * (ops.exp(x / alpha) - 1.0), zero)


def selu(x: object) -> object:
    """Computes the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    pos = ops.maximum(x, ops.full_like(x, 0.0))
    neg = ops.multiply(ops.full_like(x, alpha), ops.expm1(ops.minimum(x, ops.full_like(x, 0.0))))
    return ops.multiply(ops.full_like(x, scale), ops.add(pos, neg))


def log_softmax(x: object, axis: object = -1) -> object:
    """Computes the logarithm of the softmax activation function.

    Args:
        x (object): The input x tensor.
        axis (object): The axis along which to perform the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler import ops

    amax = ops.max(x, axis=axis, keepdims=True)
    shifted = ops.subtract(x, amax)
    sum_exp = ops.sum(ops.exp(shifted), axis=axis, keepdims=True)
    return ops.subtract(shifted, ops.log(sum_exp))


def glu(x: object, axis: int = -1) -> object:
    """Computes the Gated Linear Unit (GLU) activation function."""
    from ml_switcheroo_compiler import ops

    a, b = ops.split(x, 2, axis=axis)
    return ops.multiply(a, sigmoid(b))


def hard_silu(x: object) -> object:
    """Computes the Hard SiLU activation function."""
    from ml_switcheroo_compiler import ops

    return ops.multiply(x, hard_sigmoid(x))


def hard_swish(x: object) -> object:
    """Computes the Hard Swish activation function."""
    from ml_switcheroo_compiler import ops

    return ops.multiply(x, ops.divide(ops.clip(ops.add(x, 3.0), 0.0, 6.0), 6.0))


def leaky_relu(x: object, negative_slope: float = 0.01) -> object:
    """Computes the Leaky ReLU activation function."""
    from ml_switcheroo_compiler import ops

    return ops.where(ops.greater_equal(x, 0.0), x, ops.multiply(x, negative_slope))


def mish(x: object) -> object:
    """Computes the Mish activation function."""
    from ml_switcheroo_compiler import ops

    return ops.multiply(x, ops.tanh(softplus(x)))


def soft_sign(x: object) -> object:
    """Computes the SoftSign activation function."""
    from ml_switcheroo_compiler import ops

    return ops.divide(x, ops.add(1.0, ops.abs(x)))


def softplus(x: object) -> object:
    """Computes the Softplus activation function."""
    from ml_switcheroo_compiler import ops

    return ops.log1p(ops.exp(x))


def sparse_plus(x: object) -> object:
    """Computes the SparsePlus activation function."""
    from ml_switcheroo_compiler import ops

    leq = ops.less_equal(x, -1.0)
    geq = ops.greater_equal(x, 1.0)
    mid = ops.multiply(0.25, ops.square(ops.add(x, 1.0)))
    return ops.where(leq, 0.0, ops.where(geq, x, mid))


def sparse_sigmoid(x: object) -> object:
    """Computes the Sparse Sigmoid activation function."""
    from ml_switcheroo_compiler import ops

    return ops.clip(ops.add(ops.multiply(0.5, x), 0.5), 0.0, 1.0)


def squareplus(x: object, b: float = 4.0) -> object:
    """Computes the SquarePlus activation function."""
    from ml_switcheroo_compiler import ops

    return ops.multiply(0.5, ops.add(x, ops.sqrt(ops.add(ops.square(x), b))))


def standardize(x: object, axis: int = -1, epsilon: float = 1e-5) -> object:
    """Standardizes the input tensor along the given axis."""
    from ml_switcheroo_compiler import ops

    mean = ops.mean(x, axis=axis, keepdims=True)
    var = ops.var(x, axis=axis, keepdims=True)
    return ops.divide(ops.subtract(x, mean), ops.sqrt(ops.add(var, epsilon)))
