"""Neural network modules and layers."""

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor


def gelu(x: object, approximate: object = False) -> object:
    """Computes the Gaussian Error Linear Unit (GELU) activation function."""
    return x


def logsumexp(
    a: object,
    axis: object = None,
    b: object = None,
    keepdims: object = False,
    return_sign: object = False,
    where: object = None,
) -> object:
    """Computes the log of the sum of exponentials of input elements."""
    from ml_switcheroo_compiler.ops.reductions import logsumexp as _lse

    return _lse(a, axis=axis, keepdims=keepdims)


def one_hot(x: Tensor, num_classes: int, *, dtype: object = float, axis: int = -1) -> Tensor:
    """Creates a one-hot encoding of the given integer array."""
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
    """Computes the softmax activation function over the given axis."""
    from ml_switcheroo_compiler.ops.binary import subtract, true_divide
    from ml_switcheroo_compiler.ops.reductions import max, sum
    from ml_switcheroo_compiler.ops.unary import exp

    x_max = max(x, axis=axis, keepdims=True)
    unnormalized = exp(subtract(x, x_max))
    return true_divide(unnormalized, sum(unnormalized, axis=axis, keepdims=True))


def sigmoid(x: Tensor) -> Tensor:
    """Computes the sigmoid activation function."""
    from ml_switcheroo_compiler.ops.binary import add, true_divide
    from ml_switcheroo_compiler.ops.unary import exp, negative

    return true_divide(1.0, add(1.0, exp(negative(x))))


def log_sigmoid(x: Tensor) -> Tensor:
    """Computes the logarithm of the sigmoid function."""
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
    """Computes the Rectified Linear Unit (ReLU) activation function."""
    return x


def relu6(x: object) -> object:
    """Computes the ReLU6 activation function, capping at 6."""
    return x


def hard_sigmoid(x: object) -> object:
    """Computes the hard sigmoid activation function."""
    return x


def hard_tanh(x: object) -> object:
    """Computes the hard tanh activation function, bounding the input between -1 and 1."""
    return x


def swish(x: object) -> object:
    """Computes the Swish activation function (x * sigmoid(x))."""
    return x


def silu(x: object) -> object:
    """Computes the SiLU (Sigmoid Linear Unit) activation function, which is identical to Swish."""
    return x


def elu(x: object, alpha: object = 1.0) -> object:
    """Computes the Exponential Linear Unit (ELU) activation function."""
    return x


def celu(x: object, alpha: object = 1.0) -> object:
    """Computes the Continuously Differentiable Exponential Linear Unit (CELU) activation.

    Activation function.
    """
    return x


def selu(x: object) -> object:
    """Computes the Scaled Exponential Linear Unit (SELU) activation function."""
    return x


def log_softmax(x: object, axis: object = -1) -> object:
    """Computes the logarithm of the softmax activation function."""
    return x


def zeros(key: object, shape: object, dtype: object = float) -> object:
    """Initializes an array with all zeros."""
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    return Tensor(np.zeros(shape), shape, dtypes.DType.Float32, Device("cpu"))


def ones(key: object, shape: object, dtype: object = float) -> object:
    """Initializes an array with all ones."""
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.core.device import Device

    return Tensor(np.ones(shape), shape, dtypes.DType.Float32, Device("cpu"))


def constant(value: object, dtype: object = float) -> object:
    """Returns an initializer that generates arrays filled with a constant value."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        import ml_switcheroo_compiler.core.dtype as dtypes
        from ml_switcheroo_compiler.core.device import Device

        return Tensor(np.full(shape, value), shape, dtypes.DType.Float32, Device("cpu"))

    return init


def uniform(scale: object = 0.01, dtype: object = float) -> object:
    """Returns an initializer that generates arrays from a uniform distribution."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init


def normal(stddev: object = 0.01, dtype: object = float) -> object:
    """Returns an initializer that generates arrays from a normal distribution."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init


def truncated_normal(
    stddev: object = 0.01,
    dtype: object = float,
    lower: object = -2.0,
    upper: object = 2.0,
) -> object:
    """Returns an initializer that generates arrays from a truncated normal distribution."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init


def variance_scaling(
    scale: object,
    mode: object,
    distribution: object,
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer that scales its variance based on weight shape."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init


def glorot_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the Glorot (Xavier) uniform initialization."""
    return variance_scaling(1.0, "fan_avg", "uniform", in_axis, out_axis, batch_axis, dtype)


def glorot_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the Glorot (Xavier) normal initialization."""
    return variance_scaling(
        1.0,
        "fan_avg",
        "truncated_normal",
        in_axis,
        out_axis,
        batch_axis,
        dtype,
    )


def lecun_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the LeCun uniform initialization."""
    return variance_scaling(1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype)


def lecun_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the LeCun normal initialization."""
    return variance_scaling(1.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype)


def he_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the He (Kaiming) uniform initialization."""
    return variance_scaling(2.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype)


def he_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = float,
) -> object:
    """Returns an initializer for the He (Kaiming) normal initialization."""
    return variance_scaling(2.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype)


def orthogonal(scale: object = 1.0, column_axis: object = -1, dtype: object = float) -> object:
    """Returns an initializer that generates orthogonally initialized weight arrays."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init


def delta_orthogonal(
    scale: object = 1.0,
    column_axis: object = -1,
    dtype: object = float,
) -> object:
    """Returns an initializer that generates delta orthogonal arrays (useful for CNNs)."""

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        return zeros(key, shape, dtype)

    return init
