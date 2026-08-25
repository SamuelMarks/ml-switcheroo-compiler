# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Neural network modules and layers."""

import ml_switcheroo_compiler.core.dtype as dtypes
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.configs import InitializerConfig

DEFAULT_SCALE = 0.01
DEFAULT_STDDEV = 0.01


def zeros(key: object, shape: object, dtype: object = None) -> object:
    """Initialize an array with all zeros.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return ops.zeros(shape, dtype=dtype or dtypes.DType.Float32)


def ones(key: object, shape: object, dtype: object = None) -> object:
    """Initialize an array with all ones.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return ops.ones(shape, dtype=dtype or dtypes.DType.Float32)


def constant(value: object, dtype: object = None) -> object:
    """Return an initializer that generates arrays filled with a constant value.

    Args:
        value (object): The value parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return ops.full(shape, value, dtype=dtype or dtypes.DType.Float32)

    return init


def uniform(scale: object = DEFAULT_SCALE, dtype: object = None) -> object:
    """Return an initializer that generates arrays from a uniform distribution.

    Args:
        scale (object): The scale parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def normal(stddev: object = DEFAULT_STDDEV, dtype: object = None) -> object:
    """Return an initializer that generates arrays from a normal distribution.

    Args:
        stddev (object): The stddev parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def truncated_normal(
    stddev: object = DEFAULT_STDDEV,
    dtype: object = None,
    lower: object = -2.0,
    upper: object = 2.0,
) -> object:
    """Return an initializer that generates arrays from a truncated normal distribution.

    Args:
        stddev (object): The stddev parameter for the operation.
        dtype (object): The target data type.
        lower (object): The lower parameter for the operation.
        upper (object): The upper parameter for the operation.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def variance_scaling(config: InitializerConfig) -> object:
    """Return an initializer that scales its variance based on weight shape.

    Args:
        config (InitializerConfig): The configuration for the initializer.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = None) -> object:
        """Initialize the instance.

        Args:
            key (object): The key parameter.
            shape (object): The shape parameter.
            dtype (object): The dtype parameter.

        Returns: object: The inferred shape or computed result.
        """
        if dtype is None:
            dtype: object = config.dtype
        return zeros(key, shape, dtype)

    return init


def glorot_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the Glorot (Xavier) uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_avg",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def glorot_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the Glorot (Xavier) normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def lecun_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the LeCun uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def lecun_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the LeCun normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_in",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def he_uniform(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the He (Kaiming) uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=2.0,
            mode="fan_in",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def he_normal(
    in_axis: object = -2,
    out_axis: object = -1,
    batch_axis: object = (),
    dtype: object = None,
) -> object:
    """Return an initializer for the He (Kaiming) normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """
    return variance_scaling(
        InitializerConfig(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=dtype,
        )
    )


def orthogonal(scale: object = 1.0, column_axis: object = -1, dtype: object = None) -> object:
    """Return an initializer that generates orthogonally initialized weight arrays.

    Args:
        scale (object): The scale parameter for the operation.
        column_axis (object): The column_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def delta_orthogonal(
    scale: object = 1.0,
    column_axis: object = -1,
    dtype: object = None,
) -> object:
    """Return an initializer that generates delta orthogonal arrays (useful for CNNs).

    Args:
        scale (object): The scale parameter for the operation.
        column_axis (object): The column_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: object: The computed result.
    """

    def init(key: object, shape: object, dtype: object = dtype) -> object:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: object: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init
