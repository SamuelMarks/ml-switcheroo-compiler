"""Provides required module functionality."""

import pytest
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    register_vjp,
    get_vjp,
    _VJP_REGISTRY,
)


def test_vjp_registry_coverage() -> None:
    """Execute the requested function."""
    if "fake_op" in _VJP_REGISTRY:
        del _VJP_REGISTRY["fake_op"]

    @register_vjp("fake_op")
    def fake_vjp() -> None:
        """Docstring."""
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_vjp("fake_op")
        def fake_vjp2() -> None:
            """Docstring."""
            pass

    assert get_vjp("fake_op") == fake_vjp
