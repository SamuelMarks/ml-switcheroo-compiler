"""Tests for test_math."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, BinaryMathOp, Digitize, Lcm, Polyadd, Polyder, Polydiv, Polyint, Polymul, Polysub, Roots, Xlog1py, Xlogy

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_binary_math_op_infer_shape() -> None:
    """Test function test_binary_math_op_infer_shape."""
    op = BinaryMathOp()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)((2, 2)) == (2, 2)


def test_digitize_infer_shape() -> None:
    """Test function test_digitize_infer_shape."""
    op = Digitize()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 2)


def test_polyder_infer_shape() -> None:
    """Test function test_polyder_infer_shape."""
    op = Polyder()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polyint_infer_shape() -> None:
    """Test function test_polyint_infer_shape."""
    op = Polyint()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polysub_infer_shape() -> None:
    """Test function test_polysub_infer_shape."""
    op = Polysub()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polyadd_infer_shape() -> None:
    """Test function test_polyadd_infer_shape."""
    op = Polyadd()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polymul_infer_shape() -> None:
    """Test function test_polymul_infer_shape."""
    op = Polymul()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polydiv_infer_shape() -> None:
    """Test function test_polydiv_infer_shape."""
    op = Polydiv()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_poly_infer_shape() -> None:
    """Test function test_poly_infer_shape."""
    from ml_switcheroo_compiler.ops.binary.math import Poly

    op = Poly()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polyval_infer_shape() -> None:
    """Test function test_polyval_infer_shape."""
    from ml_switcheroo_compiler.ops.binary.math import Polyval

    op = Polyval()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_polyfit_infer_shape() -> None:
    """Test function test_polyfit_infer_shape."""
    from ml_switcheroo_compiler.ops.binary.math import Polyfit

    op = Polyfit()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_roots_infer_shape() -> None:
    """Test function test_roots_infer_shape."""
    op = Roots()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_lcm_infer_shape() -> None:
    """Test function test_lcm_infer_shape."""
    op = Lcm()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(1) == 1


def test_array_equiv_infer_shape() -> None:
    """Test function test_array_equiv_infer_shape."""
    op = ArrayEquiv()
    assert op.infer_shape() == ()


def test_xlogy_infer_shape() -> None:
    """Test function test_xlogy_infer_shape."""
    op = Xlogy()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)


def test_xlog1py_infer_shape() -> None:
    """Test function test_xlog1py_infer_shape."""
    op = Xlog1py()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()), "shape")) == (2, 2)


def test_clip_eager() -> None:
    """Test function test_clip_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.binary.math import Clip, clip

    op = Clip()
    assert op.infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 2)

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 2)
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        import numpy as np

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        class MockBackend:
            """Test class MockBackend."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return "eager_clip"

        orig_backend = config.backend
        config.backend = "mock_clip"
        cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_clip", MockBackend)
        assert clip(np.array([1.0]), 0.0, 2.0) == "eager_clip"
        config.eager_mode = False
        clip(np.array([1.0]), 0.0, 2.0)
    finally:
        config.eager_mode = orig_eager
        if "orig_backend" in locals():
            config.backend = orig_backend


def test_rem_op() -> None:
    """Test function test_rem_op."""
    from ml_switcheroo_compiler.ops.binary.math import rem

    with pytest.raises(TypeError):
        rem(1, 2)


def test_igamma_grad_a() -> None:
    """Test function test_igamma_grad_a."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.binary.math import igamma_grad_a

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert igamma_grad_a(1, 2, kw=3) == "mock_result"
        mock_dispatch.assert_called_once_with("IgammaGradA", 1, 2, kw=3)


def test_random_gamma_grad() -> None:
    """Test function test_random_gamma_grad."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.binary.math import random_gamma_grad

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert random_gamma_grad(1, 2, kw=3) == "mock_result"
        mock_dispatch.assert_called_once_with("RandomGammaGrad", 1, 2, kw=3)


def test_sort_key_val() -> None:
    """Test function test_sort_key_val."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.binary.math import sort_key_val

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert sort_key_val(1, 2, kw=3) == "mock_result"
        mock_dispatch.assert_called_once_with("SortKeyVal", 1, 2, kw=3)


def test_missing_binary_math_coverage() -> None:
    """Test function test_missing_binary_math_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.binary.math import igamma_grad_a, random_gamma_grad, sort_key_val

    t = Tensor(np.array(1.0), TensorConfig((), "float32", "cpu"))
    with pytest.raises(RuntimeError):
        igamma_grad_a(t, t)
    with pytest.raises(RuntimeError):
        random_gamma_grad(t, t)
    with pytest.raises(RuntimeError):
        sort_key_val(t, t)
