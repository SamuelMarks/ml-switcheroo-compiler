"""Module docstring."""

import pytest
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


def test_base_generator_not_implemented() -> None:
    """Docstring."""
    with pytest.raises(NotImplementedError):
        BaseGenerator.execute_op("Op")
    with pytest.raises(NotImplementedError):
        BaseGenerator.zeros((2, 2))
    with pytest.raises(NotImplementedError):
        BaseGenerator.array([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.asarray([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.item(1)
