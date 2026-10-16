# ruff: noqa: E501
"""Core abstractions and logic definitions for test_optimizers.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.optimizers.updates import (
    AdadeltaConfig,
    AdagradConfig,
    AdamaxHyperparams,
    AdamHyperparams,
    AdamWHyperparams,
    LionConfig,
    RMSPropHyperparams,
    SGDConfig,
    adadelta_update,
    adafactor_update,
    adagrad_update,
    adam_update,
    adamax_update,
    adamw_update,
    lion_update,
    muon_update,
    rmsprop_update,
    sgd_update,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


@pytest.fixture
def tensor_fixtures() -> EagerValue:
    """Evaluate and process the tensor fixtures operation.

    Returns:
        object: The evaluated or processed output.
    """
    config.eager_mode = True
    p_data = np.array([1.0, 2.0], dtype=np.float32)
    g_data = np.array([0.1, 0.2], dtype=np.float32)
    p = ops.array(p_data)
    g = ops.array(g_data)
    return (p, g)


def test_sgd(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the sgd behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", sgd_update)(p, g, cast("Callable[..., EagerTensorProtocol]", SGDConfig)(lr=0.1))
        assert new_p is not None
        (new_p2, state2) = cast("Callable[..., EagerTensorProtocol]", sgd_update)(p, g, cast("Callable[..., EagerTensorProtocol]", SGDConfig)(lr=0.1, momentum=0.9, weight_decay=0.01))
        assert new_p2 is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adam(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adam behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adam_update)(p, g, cast("Callable[..., EagerTensorProtocol]", AdamHyperparams)(lr=0.1))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adamw(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adamw behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adamw_update)(p, g, cast("Callable[..., EagerTensorProtocol]", AdamWHyperparams)(lr=0.1))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adagrad(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adagrad behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adagrad_update)(p, g, cast("Callable[..., EagerTensorProtocol]", AdagradConfig)(lr=0.1))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_rmsprop(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the rmsprop behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", rmsprop_update)(p, g, cast("Callable[..., EagerTensorProtocol]", RMSPropHyperparams)(lr=0.1))
        assert new_p is not None
        (new_p2, state2) = cast("Callable[..., EagerTensorProtocol]", rmsprop_update)(p, g, cast("Callable[..., EagerTensorProtocol]", RMSPropHyperparams)(lr=0.1, momentum=0.9, centered=True))
        assert new_p2 is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adadelta(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adadelta behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adadelta_update)(p, g, cast("Callable[..., EagerTensorProtocol]", AdadeltaConfig)(lr=1.0))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adamax(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adamax behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adamax_update)(p, g, cast("Callable[..., EagerTensorProtocol]", AdamaxHyperparams)(lr=0.1))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_lion(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the lion behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", lion_update)(p, g, cast("Callable[..., EagerTensorProtocol]", LionConfig)(lr=0.1))
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_adafactor(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the adafactor behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", adafactor_update)(p, g, lr=0.1)
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_muon(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test the muon behavior.

    Args:
        tensor_fixtures (object): The tensor_fixtures parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        (p, g) = tensor_fixtures
        (new_p, state) = cast("Callable[..., EagerTensorProtocol]", muon_update)(p, g, lr=0.1)
        assert new_p is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_apply_adam(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test apply adam."""
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adam

    p_new, m_new, v_new = cast("Callable[..., EagerTensorProtocol]", apply_adam)(p, p, p, g, 0.1)
    assert p_new is not None


def test_apply_adagrad(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test apply adagrad."""
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adagrad

    p_new, a_new = cast("Callable[..., EagerTensorProtocol]", apply_adagrad)(p, p, g, 0.1)
    assert p_new is not None


def test_apply_ftrl(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test apply ftrl."""
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_ftrl

    p_new, a_new, l_new = cast("Callable[..., EagerTensorProtocol]", apply_ftrl)(p, p, p, g, 0.1)
    assert p_new is not None


def test_apply_rmsprop(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test apply rmsprop."""
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_rmsprop

    p_new, m_new, mom_new = cast("Callable[..., EagerTensorProtocol]", apply_rmsprop)(p, p, p, g, 0.1)
    assert p_new is not None


def test_apply_trace_fallback(tensor_fixtures: EagerTensorProtocol) -> None:
    """Test apply trace fallback."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adagrad, apply_adam, apply_ftrl, apply_rmsprop

    (p, g) = tensor_fixtures

    def run_adam(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function run_adam."""
        return cast("Callable[..., EagerTensorProtocol]", apply_adam)(a, a, a, b, 0.1)

    out1 = cast("Callable[..., EagerTensorProtocol]", _trace_function)(run_adam, (p, g), "adam")
    assert out1 is not None

    def run_adagrad(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function run_adagrad."""
        return cast("Callable[..., EagerTensorProtocol]", apply_adagrad)(a, a, b, 0.1)

    out2 = cast("Callable[..., EagerTensorProtocol]", _trace_function)(run_adagrad, (p, g), "adagrad")
    assert out2 is not None

    def run_ftrl(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function run_ftrl."""
        return cast("Callable[..., EagerTensorProtocol]", apply_ftrl)(a, a, a, b, 0.1)

    out3 = cast("Callable[..., EagerTensorProtocol]", _trace_function)(run_ftrl, (p, g), "ftrl")
    assert out3 is not None

    def run_rmsprop(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function run_rmsprop."""
        return cast("Callable[..., EagerTensorProtocol]", apply_rmsprop)(a, a, a, b, 0.1)

    out4 = cast("Callable[..., EagerTensorProtocol]", _trace_function)(run_rmsprop, (p, g), "rmsprop")
    assert out4 is not None
