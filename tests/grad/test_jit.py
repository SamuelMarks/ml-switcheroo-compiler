"""Behavioral and 100% branch-coverage unit tests for `jit`, `disable_jit`, and symbolic `eval_shape`."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import TensorSpec

import ml_switcheroo_compiler.grad as grad_pkg
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad
from ml_switcheroo_compiler.grad.jit import (
    JitOptions,
    clear_jit_cache,
    disable_jit,
    eval_shape,
    get_jit_cache_size,
    is_jit_disabled,
    jit,
    set_max_jit_cache_entries,
)
from ml_switcheroo_compiler.grad.options import JitOptions as LegacyJitOptions
from ml_switcheroo_compiler.tracing.tracer import get_trace_count, reset_trace_count
from tests.testing_utils import adapt_backend_module

jit_mod = importlib.import_module("ml_switcheroo_compiler.grad.jit")
_ = (
    JitOptions,
    clear_jit_cache,
    disable_jit,
    eval_shape,
    get_jit_cache_size,
    is_jit_disabled,
    jit,
    set_max_jit_cache_entries,
)


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair `(a, b)`.
    """
    return (a, b)


def test_grad_init_and_jit_scalar_passthrough() -> None:
    """Verify grad package exports, JitOptions defaults, scalar passthrough, and disable_jit."""
    loaded_mod = importlib.import_module("ml_switcheroo_compiler.grad.jit")

    assert adapt_backend_module(grad_pkg).has_symbol("grad")
    assert adapt_backend_module(grad_pkg).has_symbol("value_and_grad")
    assert adapt_backend_module(grad_pkg).has_symbol("jit")
    assert adapt_backend_module(grad_pkg).has_symbol("disable_jit")
    assert adapt_backend_module(grad_pkg).has_symbol("eval_shape")
    assert "grad" in grad_pkg.__all__
    assert "jit" in grad_pkg.__all__

    opts = loaded_mod.JitOptions()
    assert opts.static_argnums is None
    assert opts.static_argnames is None
    assert opts.donate_argnums is None
    assert opts.donate_argnames is None
    assert opts.keep_unused is False
    assert opts.device is None
    assert opts.backend is None
    assert opts.inline is False
    assert opts.abstracted_axes is None

    jitted1 = loaded_mod.jit(_sample_add_fn)
    assert jitted1(5, y=15) == 20

    jitted2 = loaded_mod.jit(_sample_add_fn, options=opts)
    assert jitted2(3) == 13

    assert not loaded_mod.is_jit_disabled()
    with loaded_mod.disable_jit():
        assert loaded_mod.is_jit_disabled()
        assert jitted1(2, y=3) == 5
    assert not loaded_mod.is_jit_disabled()

    assert loaded_mod.eval_shape(_sample_shape_fn, 3, 4) == (3, 4)


def test_jit_compilation_caching_unused_pruning_and_options() -> None:
    """Verify JIT trace caching, PassManager dead-input pruning, static args, and LRU eviction."""
    jit_mod.clear_jit_cache()
    assert jit_mod.get_jit_cache_size() == 0

    def _poly(x: Tensor, unused: Tensor, scale: int = 2, mode: str = "dbl") -> tuple[Tensor, Tensor, str]:
        """Compute polynomial with duplicate output node and literal metadata.

        Args:
            x (Tensor): Primary dynamic tensor.
            unused (Tensor): Unused dynamic tensor for dead-input pruning test.
            scale (int): Dynamic/static scalar factor.
            mode (str): Static mode label.

        Returns:
            tuple[Tensor, Tensor, str]: `(out, out, mode)`.
        """
        _ = unused
        base = x + x if mode == "dbl" else x * x
        out = base * scale
        return (out, out, mode)

    reset_trace_count(_poly)
    opts = jit_mod.JitOptions(
        static_argnums=2,
        static_argnames="mode",
        keep_unused=False,
        backend="numpy",
        device="cpu",
    )
    compiled = jit_mod.jit(_poly, options=opts)

    x1 = Tensor(np.array([1.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))
    u1 = Tensor(np.array([9.0, 9.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))

    res1 = compiled(x1, u1, 3, mode="dbl")
    assert isinstance(res1, tuple) and len(res1) == 3
    assert isinstance(res1[0], Tensor)
    assert np.allclose(np.asarray(res1[0].data), np.array([6.0, 18.0], dtype=np.float32))
    assert res1[2] == "dbl"
    assert get_trace_count(_poly) == 1
    assert jit_mod.get_jit_cache_size() == 1

    # Cache hit on second call with identical shape/dtype and static args
    x2 = Tensor(np.array([2.0, 5.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))
    res2 = compiled(x2, u1, 3, mode="dbl")
    assert isinstance(res2, tuple) and isinstance(res2[0], Tensor)
    assert np.allclose(np.asarray(res2[0].data), np.array([12.0, 30.0], dtype=np.float32))
    assert get_trace_count(_poly) == 1

    # Changing static_argnames triggers re-trace
    res3 = compiled(x2, u1, 3, mode="sq")
    assert isinstance(res3, tuple) and isinstance(res3[0], Tensor)
    assert np.allclose(np.asarray(res3[0].data), np.array([12.0, 75.0], dtype=np.float32))
    assert get_trace_count(_poly) == 2

    # Sequence static_argnums/static_argnames, keep_unused=True, non-Tensor dynamic leaf, and LRU eviction
    opts_seq = jit_mod.JitOptions(
        static_argnums=(2,),
        static_argnames=("mode",),
        keep_unused=True,
    )
    with pytest.raises(ValueError, match="limit must be >= 1"):
        set_max_jit_cache_entries(0)

    compiled_keep = jit_mod.jit(lambda x, k, s, mode="a": x + k, options=opts_seq)
    saved_max = set_max_jit_cache_entries(1)
    try:
        out_k = compiled_keep(x1, 1, 0, mode="a")
        assert isinstance(out_k, Tensor)
        assert jit_mod.get_jit_cache_size() == 1
    finally:
        set_max_jit_cache_entries(saved_max)

    # Default JitOptions() on Tensor function (covers None static_argnums/static_argnames/backend/device)
    default_tensor_jit = jit_mod.jit(lambda t: t + t, options=jit_mod.JitOptions())
    out_def = default_tensor_jit(x1)
    assert isinstance(out_def, Tensor)

    # Legacy grad.options.JitOptions + inline=True
    legacy_inline = jit_mod.jit(lambda t: t + t, options=LegacyJitOptions(inline=True))
    out_inl = legacy_inline(x1)
    assert isinstance(out_inl, Tensor)


def test_jit_and_grad_bidirectional_composition() -> None:
    """Verify both `grad(jit(f))` and `jit(grad(f))` compose through the unified LogicalGraph IR."""

    def _cubic(x: Tensor) -> Tensor:
        """Compute `x * x * x` for derivative composition check.

        Args:
            x (Tensor): Scalar tensor input.

        Returns:
            Tensor: Cubic scalar tensor.
        """
        return x * x * x

    x = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu", requires_grad=True))

    g_of_jit = grad(jit_mod.jit(_cubic))(x)
    jit_of_g = jit_mod.jit(grad(_cubic))(x)
    assert isinstance(g_of_jit, Tensor) and isinstance(jit_of_g, Tensor)
    assert float(np.asarray(g_of_jit.data)) == pytest.approx(12.0)
    assert float(np.asarray(jit_of_g.data)) == pytest.approx(12.0)


def test_eval_shape_zero_allocation_symbolic_inference() -> None:
    """Verify `eval_shape` infers output TensorSpecs symbolically without eager array allocation."""

    def _net(x: Tensor, w: Tensor, bias: Tensor, label: str = "ok") -> dict[str, OpValue]:
        """Symbolic linear + elementwise network returning a PyTree dictionary.

        Args:
            x (Tensor): Input activations `(4, 3)`.
            w (Tensor): Weight matrix `(3, 5)`.
            bias (Tensor): Bias vector `(4, 5)`.
            label (str): Non-tensor PyTree leaf.

        Returns:
            dict[str, OpValue]: Output dictionary with tensor and metadata leaves.
        """
        from ml_switcheroo_compiler.ops.linalg import matmul

        y = matmul(x, w) + bias
        return {"out": y, "tag": label}

    x_spec = TensorSpec(shape=(4, 3), dtype="float32")
    w_cfg = TensorConfig((3, 5), DType.Float32, "cpu")
    b_tensor = Tensor(np.zeros((4, 5), dtype=np.float32), TensorConfig((4, 5), DType.Float32, "cpu"))

    with patch(
        "ml_switcheroo_compiler.ops.eager_evaluator.EagerEvaluator.evaluate",
        side_effect=AssertionError("eval_shape must not invoke eager kernels"),
    ):
        inferred = jit_mod.eval_shape(_net, x_spec, w_cfg, b_tensor, label="symbolic")

    assert isinstance(inferred, dict)
    out_spec = inferred["out"]
    assert isinstance(out_spec, TensorSpec)
    assert out_spec.shape == (4, 5)
    assert out_spec.dtype == "float32"
    assert inferred["tag"] == "symbolic"
