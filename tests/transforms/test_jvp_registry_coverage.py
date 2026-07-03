"""Provides required module functionality."""

import pytest

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import (
    _JVP_REGISTRY,
    get_jvp,
    register_jvp,
)


def test_jvp_registry_coverage_brute() -> None:
    """Execute the requested function."""
    if "fake_op" in _JVP_REGISTRY:
        del _JVP_REGISTRY["fake_op"]

    @register_jvp("fake_op")
    def fake_jvp() -> None:
        """Docstring."""

    assert get_jvp("fake_op") == fake_jvp

    with pytest.raises(ValueError, match="already registered"):

        @register_jvp("fake_op")
        def fake_jvp2() -> None:
            """Docstring."""

    with pytest.raises(UnimplementedMathError):
        get_jvp("non_existent_op_fake_xyz")
