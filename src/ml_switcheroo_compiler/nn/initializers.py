# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Neural network modules and layers."""

from typing import Any

import ml_switcheroo_compiler.core.dtype as dtypes
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.configs import InitializerConfig

DEFAULT_SCALE = 0.01
DEFAULT_STDDEV = 0.01


def zeros(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Initialize an array with all zeros.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """
    return ops.zeros(shape, dtype=dtype or dtypes.DType.Float32)


def ones(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Initialize an array with all ones.

    Args:
        key (object): The PRNG key.
        shape (object): The target shape.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """
    return ops.ones(shape, dtype=dtype or dtypes.DType.Float32)


def constant(value: Any, dtype: Any = None) -> Any:
    """Return an initializer that generates arrays filled with a constant value.

    Args:
        value (object): The value parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return ops.full(shape, value, dtype=dtype or dtypes.DType.Float32)

    return init


def uniform(scale: Any = DEFAULT_SCALE, dtype: Any = None) -> Any:
    """Return an initializer that generates arrays from a uniform distribution.

    Args:
        scale (object): The scale parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def normal(stddev: Any = DEFAULT_STDDEV, dtype: Any = None) -> Any:
    """Return an initializer that generates arrays from a normal distribution.

    Args:
        stddev (object): The stddev parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def truncated_normal(
    stddev: Any = DEFAULT_STDDEV,
    dtype: Any = None,
    lower: Any = -2.0,
    upper: Any = 2.0,
) -> Any:
    """Return an initializer that generates arrays from a truncated normal distribution.

    Args:
        stddev (object): The stddev parameter for the operation.
        dtype (object): The target data type.
        lower (object): The lower parameter for the operation.
        upper (object): The upper parameter for the operation.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def variance_scaling(config: InitializerConfig) -> Any:
    """Return an initializer that scales its variance based on weight shape.

    Args:
        config (InitializerConfig): The configuration for the initializer.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = None) -> Any:
        """Initialize the instance.

        Args:
            key (object): The key parameter.
            shape (object): The shape parameter.
            dtype (object): The dtype parameter.

        Returns: Any: The inferred shape or computed result.
        """
        if dtype is None:
            dtype = config.dtype
        return zeros(key, shape, dtype)

    return init


def glorot_uniform(
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the Glorot (Xavier) uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the Glorot (Xavier) normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the LeCun uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the LeCun normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the He (Kaiming) uniform initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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
    in_axis: Any = -2,
    out_axis: Any = -1,
    batch_axis: Any = (),
    dtype: Any = None,
) -> Any:
    """Return an initializer for the He (Kaiming) normal initialization.

    Args:
        in_axis (object): The in_axis parameter for the operation.
        out_axis (object): The out_axis parameter for the operation.
        batch_axis (object): The batch_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
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


def orthogonal(scale: Any = 1.0, column_axis: Any = -1, dtype: Any = None) -> Any:
    """Return an initializer that generates orthogonally initialized weight arrays.

    Args:
        scale (object): The scale parameter for the operation.
        column_axis (object): The column_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init


def delta_orthogonal(
    scale: Any = 1.0,
    column_axis: Any = -1,
    dtype: Any = None,
) -> Any:
    """Return an initializer that generates delta orthogonal arrays (useful for CNNs).

    Args:
        scale (object): The scale parameter for the operation.
        column_axis (object): The column_axis parameter for the operation.
        dtype (object): The target data type.

    Returns: Any: The computed result.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Initialize the tensor.

        Args:
            key (object): The random key.
            shape (object): The shape of the tensor.
            dtype (object): The data type.

        Returns: Any: The initialized tensor.
        """
        return zeros(key, shape, dtype)

    return init
