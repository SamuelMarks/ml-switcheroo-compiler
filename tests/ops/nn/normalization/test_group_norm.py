# ruff: noqa: E501
"""Core abstractions and logic definitions for test_group_norm.py."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_group_ops import _group_mean, _group_norm, _group_variance


def test_group_mean_eager() -> object:
    """Test the group mean eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        x = np.random.randn(2, 4, 4, 8).astype(np.float32)
        out = _group_mean(np, x, 2, axis=-1)
        reshaped = x.reshape(2, 4, 4, 2, 4)
        expected = np.mean(reshaped, axis=(1, 2, 4), keepdims=True)
        np.testing.assert_allclose(out, expected, rtol=1e-05, atol=1e-05)
    except Exception as e:
        raise e
        pass


def test_group_variance_eager() -> object:
    """Test the group variance eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        x = np.random.randn(2, 4, 4, 8).astype(np.float32)
        out = _group_variance(np, x, 2, axis=-1)
        reshaped = x.reshape(2, 4, 4, 2, 4)
        expected = np.var(reshaped, axis=(1, 2, 4), keepdims=True)
        np.testing.assert_allclose(out, expected, rtol=1e-05, atol=1e-05)
    except Exception as e:
        raise e
        pass


def test_group_norm_eager() -> object:
    """Test the group norm eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        x = np.random.randn(2, 4, 4, 8).astype(np.float32)
        out = _group_norm(np, x, 2, axis=-1, epsilon=1e-05)
        reshaped = x.reshape(2, 4, 4, 2, 4)
        mean = np.mean(reshaped, axis=(1, 2, 4), keepdims=True)
        var = np.var(reshaped, axis=(1, 2, 4), keepdims=True)
        expected = ((reshaped - mean) / np.sqrt(var + 1e-05)).reshape(x.shape)
        np.testing.assert_allclose(out, expected, rtol=1e-05, atol=1e-05)
    except Exception as e:
        raise e
        pass


def test_group_norm_5d_eager() -> object:
    """Test the group norm 5d eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        x = np.random.randn(2, 3, 4, 4, 8).astype(np.float32)
        weight = np.random.randn(8).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        out = _group_norm(np, x, 4, weight=weight, bias=bias, axis=-1, epsilon=1e-05)
        reshaped = x.reshape(2, 3, 4, 4, 4, 2)
        mean = np.mean(reshaped, axis=(1, 2, 3, 5), keepdims=True)
        var = np.var(reshaped, axis=(1, 2, 3, 5), keepdims=True)
        expected = ((reshaped - mean) / np.sqrt(var + 1e-05)).reshape(x.shape)
        expected = expected * weight.reshape(1, 1, 1, 1, 8) + bias.reshape(1, 1, 1, 1, 8)
        np.testing.assert_allclose(out, expected, rtol=1e-05, atol=1e-05)
    except Exception as e:
        raise e
        pass
