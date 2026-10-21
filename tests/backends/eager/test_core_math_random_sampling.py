"""Tests for test_core_math_random_sampling."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_random_sampling import (
    _np_gumbel,
    _np_rnguniform,
    _np_wald,
    _pshuffle,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_random_sampling_coverage() -> None:
    """Test all branches and lines of math_random_sampling.

    Returns:
        None
    """

    # 1. _pshuffle
    # Case A: backend has lax.pshuffle
    class LaxObj:
        """Mock lax container."""

        def pshuffle(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock pshuffle.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "pshuffle_val"

    class BackendWithLax:
        """Mock backend with lax."""

        lax = LaxObj()

    assert _pshuffle(as_mod(BackendWithLax()), [1, 2, 3]) == "pshuffle_val"

    # Case B: backend lacks lax -> returns args[0] if args else None
    class EmptyBackend:
        """Empty backend."""

        pass

    assert _pshuffle(as_mod(EmptyBackend()), [1, 2, 3]) == [1, 2, 3]
    assert _pshuffle(as_mod(EmptyBackend())) is None

    # 2. _np_gumbel
    # Case A: backend has gumbel
    class GumbelBackend:
        """Backend with gumbel."""

        def gumbel(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock gumbel.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "gumbel_val"

    assert _np_gumbel(as_mod(GumbelBackend()), 0.0, 1.0) == "gumbel_val"

    # Case B: backend lacks gumbel -> fallback to np.random.gumbel
    g_res = _np_gumbel(as_mod(EmptyBackend()), 0.0, 1.0, size=(2, 2))
    assert as_tensor(getattr(g_res, "shape")) == (2, 2)

    # 3. _np_rnguniform
    # Case A: backend has rnguniform
    class RngUniformBackend:
        """Backend with rnguniform."""

        def rnguniform(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock rnguniform.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "rnguniform_val"

    assert _np_rnguniform(as_mod(RngUniformBackend()), 0.0, 1.0) == "rnguniform_val"

    # Case B: backend lacks rnguniform -> fallback to np.random.uniform
    u_res = _np_rnguniform(as_mod(EmptyBackend()), 0.0, 1.0, size=(3,))
    assert as_tensor(getattr(u_res, "shape")) == (3,)

    # 4. _np_wald
    # Case A: backend has wald
    class WaldBackend:
        """Backend with wald."""

        def wald(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock wald.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "wald_val"

    assert _np_wald(as_mod(WaldBackend()), 1.0, 1.0) == "wald_val"

    # Case B: backend lacks wald -> fallback to np.random.wald
    w_res = _np_wald(as_mod(EmptyBackend()), 1.0, 1.0, size=(2,))
    assert as_tensor(getattr(w_res, "shape")) == (2,)

    # Registry verification
    assert global_eager_registry.get("Pshuffle") is not None
    assert global_eager_registry.get("Gumbel") is not None
    assert global_eager_registry.get("RngUniform") is not None
    assert global_eager_registry.get("Wald") is not None
