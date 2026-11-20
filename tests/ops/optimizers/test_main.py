"""Core abstractions and logic definitions for test_optimizers.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
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
from tests.testing_utils import ensure_tensor, extract_test_shape, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


@pytest.fixture
def tensor_fixtures() -> tuple[Tensor, Tensor]:
    """Evaluate and process the tensor fixtures operation.

    Returns:
        tuple[Tensor, Tensor]: The evaluated or processed output.
    """
    config.eager_mode = True
    p_data = np.array([1.0, 2.0], dtype=np.float32)
    g_data = np.array([0.1, 0.2], dtype=np.float32)
    p = ensure_tensor(ops.array(p_data))
    g = ensure_tensor(ops.array(g_data))
    return (p, g)


def test_sgd(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the sgd behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = sgd_update(p, g, SGDConfig(lr=0.1))
    assert tuple(new_p.shape) == (2,)
    (new_p2, state2) = sgd_update(p, g, SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.01))
    assert tuple(new_p2.shape) == (2,)


def test_adam(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adam behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adam_update(p, g, AdamHyperparams(lr=0.1))
    assert tuple(new_p.shape) == (2,)


def test_adamw(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adamw behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adamw_update(p, g, AdamWHyperparams(lr=0.1))
    assert tuple(new_p.shape) == (2,)


def test_adagrad(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adagrad behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adagrad_update(p, g, AdagradConfig(lr=0.1))
    assert tuple(new_p.shape) == (2,)


def test_rmsprop(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the rmsprop behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = rmsprop_update(p, g, RMSPropHyperparams(lr=0.1))
    assert tuple(new_p.shape) == (2,)
    (new_p2, state2) = rmsprop_update(p, g, RMSPropHyperparams(lr=0.1, momentum=0.9, centered=True))
    assert tuple(new_p2.shape) == (2,)


def test_adadelta(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adadelta behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adadelta_update(p, g, AdadeltaConfig(lr=1.0))
    assert tuple(new_p.shape) == (2,)


def test_adamax(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adamax behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adamax_update(p, g, AdamaxHyperparams(lr=0.1))
    assert tuple(new_p.shape) == (2,)


def test_lion(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the lion behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = lion_update(p, g, LionConfig(lr=0.1))
    assert tuple(new_p.shape) == (2,)


def test_adafactor(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the adafactor behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = adafactor_update(p, g, lr=0.1)
    assert tuple(new_p.shape) == (2,)


def test_muon(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test the muon behavior.

    Args:
        tensor_fixtures: Parameter and gradient fixture pair.
    """
    (p, g) = tensor_fixtures
    (new_p, state) = muon_update(p, g, lr=0.1)
    assert tuple(new_p.shape) == (2,)


def test_apply_adam(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test apply adam.

    Args:
        tensor_fixtures: Test parameter `tensor_fixtures`.
    """
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adam
    from tests.testing_utils import coerce_operand_list

    (p_new, m_new, v_new) = coerce_operand_list(apply_adam(p, p, p, g, 0.1))
    assert tuple(extract_test_shape(p_new)) == (2,)


def test_apply_adagrad(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test apply adagrad.

    Args:
        tensor_fixtures: Test parameter `tensor_fixtures`.
    """
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adagrad

    (p_new, a_new) = unpack_pair(apply_adagrad(p, p, g, 0.1))
    assert tuple(extract_test_shape(p_new)) == (2,)


def test_apply_ftrl(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test apply ftrl.

    Args:
        tensor_fixtures: Test parameter `tensor_fixtures`.
    """
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_ftrl
    from tests.testing_utils import coerce_operand_list

    (p_new, a_new, l_new) = coerce_operand_list(apply_ftrl(p, p, p, g, 0.1))
    assert tuple(extract_test_shape(p_new)) == (2,)


def test_apply_rmsprop(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test apply rmsprop.

    Args:
        tensor_fixtures: Test parameter `tensor_fixtures`.
    """
    (p, g) = tensor_fixtures
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_rmsprop
    from tests.testing_utils import coerce_operand_list

    (p_new, m_new, mom_new) = coerce_operand_list(apply_rmsprop(p, p, p, g, 0.1))
    assert tuple(extract_test_shape(p_new)) == (2,)


def test_apply_trace_fallback(tensor_fixtures: tuple[Tensor, Tensor]) -> None:
    """Test apply trace fallback.

    Args:
        tensor_fixtures: Test parameter `tensor_fixtures`.
    """
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
    from ml_switcheroo_compiler.ops.optimizers.updates import apply_adagrad, apply_adam, apply_ftrl, apply_rmsprop

    (p, g) = tensor_fixtures

    def run_adam(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function run_adam.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `run_adam`.
        """
        ta, tb = ensure_tensor(a), ensure_tensor(b)
        return apply_adam(ta, ta, ta, tb, 0.1)

    out1 = _trace_function(run_adam, (p, g), "adam")
    assert type(out1).__name__ == "SubgraphWithId"

    def run_adagrad(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function run_adagrad.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `run_adagrad`.
        """
        ta, tb = ensure_tensor(a), ensure_tensor(b)
        return apply_adagrad(ta, ta, tb, 0.1)

    out2 = _trace_function(run_adagrad, (p, g), "adagrad")
    assert type(out2).__name__ == "SubgraphWithId"

    def run_ftrl(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function run_ftrl.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `run_ftrl`.
        """
        ta, tb = ensure_tensor(a), ensure_tensor(b)
        return apply_ftrl(ta, ta, ta, tb, 0.1)

    out3 = _trace_function(run_ftrl, (p, g), "ftrl")
    assert type(out3).__name__ == "SubgraphWithId"

    def run_rmsprop(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function run_rmsprop.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `run_rmsprop`.
        """
        ta, tb = ensure_tensor(a), ensure_tensor(b)
        return apply_rmsprop(ta, ta, ta, tb, 0.1)

    out4 = _trace_function(run_rmsprop, (p, g), "rmsprop")
    assert type(out4).__name__ == "SubgraphWithId"
