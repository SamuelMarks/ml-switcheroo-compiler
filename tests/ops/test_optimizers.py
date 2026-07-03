"""Module docstring."""

import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.optimizers.updates import (
    AdamaxHyperparams,
    AdamHyperparams,
    AdamWHyperparams,
    RMSPropHyperparams,
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


@pytest.fixture
def tensor_fixtures() -> object:
    """Function docstring."""
    config.eager_mode = True
    p_data = np.array([1.0, 2.0], dtype=np.float32)
    g_data = np.array([0.1, 0.2], dtype=np.float32)
    p = ops.array(p_data)
    g = ops.array(g_data)
    return p, g


def test_sgd(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = sgd_update(p, g, lr=0.1)
    assert new_p is not None

    new_p2, state2 = sgd_update(p, g, lr=0.1, momentum=0.9, weight_decay=0.01)
    assert new_p2 is not None


def test_adam(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adam_update(p, g, AdamHyperparams(lr=0.1))
    assert new_p is not None


def test_adamw(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adamw_update(p, g, AdamWHyperparams(lr=0.1))
    assert new_p is not None


def test_adagrad(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adagrad_update(p, g, lr=0.1)
    assert new_p is not None


def test_rmsprop(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = rmsprop_update(p, g, RMSPropHyperparams(lr=0.1))
    assert new_p is not None

    new_p2, state2 = rmsprop_update(p, g, RMSPropHyperparams(lr=0.1, momentum=0.9, centered=True))
    assert new_p2 is not None


def test_adadelta(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adadelta_update(p, g, lr=1.0)
    assert new_p is not None


def test_adamax(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adamax_update(p, g, AdamaxHyperparams(lr=0.1))
    assert new_p is not None


def test_lion(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = lion_update(p, g, lr=0.1)
    assert new_p is not None


def test_adafactor(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = adafactor_update(p, g, lr=0.1)
    assert new_p is not None


def test_muon(tensor_fixtures: object) -> object:
    """Function docstring."""
    p, g = tensor_fixtures
    new_p, state = muon_update(p, g, lr=0.1)
    assert new_p is not None
