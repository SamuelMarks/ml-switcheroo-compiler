"""Provides required module functionality."""

import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    _VJP_REGISTRY,
    get_vjp,
    register_vjp,
)


def test_vjp_registry_coverage() -> None:
    """Execute the requested function."""
    if "fake_op" in _VJP_REGISTRY:
        del _VJP_REGISTRY["fake_op"]

    @register_vjp("fake_op")
    def fake_vjp() -> None:
        """Docstring."""

    with pytest.raises(ValueError, match="already registered"):

        @register_vjp("fake_op")
        def fake_vjp2() -> None:
            """Docstring."""

    assert get_vjp("fake_op") == fake_vjp
