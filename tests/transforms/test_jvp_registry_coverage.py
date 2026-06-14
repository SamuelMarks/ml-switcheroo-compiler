"""Provides required module functionality."""

import pytest
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import (
    register_jvp,
    get_jvp,
    _JVP_REGISTRY,
)


def test_jvp_registry_coverage_brute() -> None:
    """Execute the requested function."""
    if "fake_op" in _JVP_REGISTRY:
        del _JVP_REGISTRY["fake_op"]

    @register_jvp("fake_op")
    def fake_jvp() -> None:
        """Docstring."""
        pass

    assert get_jvp("fake_op") == fake_jvp

    with pytest.raises(ValueError, match="already registered"):

        @register_jvp("fake_op")
        def fake_jvp2() -> None:
            """Docstring."""
            pass

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        get_jvp("non_existent_op_fake_xyz")
