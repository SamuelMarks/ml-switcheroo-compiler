"""Profiler synthetic backend module installers for isolated test environments."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.backend_stubs import (
    _MockJaxArray,
    _MockNNLinear,
    _MockNNModule,
    _MockTorchDevice,
    _MockTorchTensor,
)


def _install_torch_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install NumPy-backed mock torch module when torch is not installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest MonkeyPatch instance.
    """
    import sys
    from unittest.mock import MagicMock

    import ml_switcheroo_compiler.backends.pytorch.eager as pt_eager
    import ml_switcheroo_compiler.backends.pytorch.profiler as pt_prof
    import ml_switcheroo_compiler.backends.pytorch.types as pt_types

    mock_torch = MagicMock()
    mock_torch.Tensor = _MockTorchTensor
    mock_torch.device = _MockTorchDevice
    mock_torch.float32 = np.float32
    mock_torch.tensor.side_effect = lambda v, dtype=np.float32, device=None: np.asarray(v, dtype=np.float32).view(
        _MockTorchTensor
    )
    mock_torch.as_tensor.side_effect = lambda v, dtype=None, device=None: np.asarray(v, dtype=np.float32).view(
        _MockTorchTensor
    )
    mock_torch.from_numpy.side_effect = lambda v: np.asarray(v).view(_MockTorchTensor)
    mock_torch.zeros.side_effect = lambda *s, **kw: np.zeros(
        s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s, dtype=np.float32
    ).view(_MockTorchTensor)
    mock_torch.ones.side_effect = lambda *s, **kw: np.ones(
        s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s, dtype=np.float32
    ).view(_MockTorchTensor)
    mock_torch.zeros_like.side_effect = lambda x, **kw: np.zeros_like(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.empty_like.side_effect = lambda x, **kw: np.empty_like(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.add.side_effect = lambda a, b: np.add(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.sub.side_effect = lambda a, b: np.subtract(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.mul.side_effect = lambda a, b: np.multiply(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.matmul.side_effect = lambda a, b: np.matmul(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockTorchTensor)
    mock_torch.exp.side_effect = lambda x: np.exp(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.sin.side_effect = lambda x: np.sin(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.sum.side_effect = lambda x, dim=None, keepdim=False: np.asarray(
        np.sum(np.asarray(x), axis=dim, keepdims=keepdim)
    ).view(_MockTorchTensor)
    mock_torch.stack.side_effect = lambda xs, dim=0: np.stack([np.asarray(t) for t in xs], axis=dim).view(
        _MockTorchTensor
    )
    mock_torch.chunk.side_effect = lambda x, chunks, dim=0: [
        t.view(_MockTorchTensor) for t in np.array_split(np.asarray(x), max(1, int(chunks)), axis=dim)
    ]
    mock_torch.nn.Module = _MockNNModule
    mock_torch.nn.Linear = _MockNNLinear
    mock_torch.nn.Parameter.side_effect = lambda x, **kw: np.asarray(x, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.nn.functional.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockTorchTensor)
    mock_torch.nn.functional.linear.side_effect = lambda x, w, b=None: (
        np.matmul(np.asarray(x), np.swapaxes(np.asarray(w), -1, -2)) + (np.asarray(b) if b is not None else 0.0)
    ).view(_MockTorchTensor)
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.mps.is_available.return_value = False
    mock_torch.compile.side_effect = lambda m, *a, **kw: m
    mock_torch.jit.trace.side_effect = lambda m, *a, **kw: m
    mock_torch.distributed.is_available.return_value = False
    mock_torch.distributed.is_initialized.return_value = False
    for mod_name, mod_obj in (
        ("torch", mock_torch),
        ("torch.nn", mock_torch.nn),
        ("torch.nn.functional", mock_torch.nn.functional),
        ("torch.distributed", mock_torch.distributed),
        ("torch.fft", mock_torch.fft),
        ("torch.linalg", mock_torch.linalg),
        ("torch.special", mock_torch.special),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    for target_mod in (pt_prof, pt_eager, pt_types):
        monkeypatch.setattr(target_mod, "torch", mock_torch)
    monkeypatch.setattr(pt_eager, "F", mock_torch.nn.functional)


def _install_jax_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install NumPy-backed mock jax module when jax is not installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest MonkeyPatch instance.
    """
    import sys
    from unittest.mock import MagicMock

    import ml_switcheroo_compiler.backends.jax.eager as jax_eager
    import ml_switcheroo_compiler.backends.jax.profiler as jax_prof
    import ml_switcheroo_compiler.backends.jax.types as jax_types

    mock_jnp = MagicMock()
    mock_jnp.ndarray = _MockJaxArray
    mock_jnp.float32 = np.float32
    mock_jnp.asarray.side_effect = lambda v, dtype=None: np.asarray(v, dtype=np.float32).view(_MockJaxArray)
    mock_jnp.array.side_effect = lambda v, dtype=None: np.array(v, dtype=np.float32).view(_MockJaxArray)
    mock_jnp.zeros.side_effect = lambda s, dtype=None: np.zeros(s, dtype=np.float32).view(_MockJaxArray)
    mock_jnp.ones.side_effect = lambda s, dtype=None: np.ones(s, dtype=np.float32).view(_MockJaxArray)
    mock_jnp.zeros_like.side_effect = lambda x, dtype=None: np.zeros_like(np.asarray(x)).view(_MockJaxArray)
    mock_jnp.add.side_effect = lambda a, b: np.add(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.subtract.side_effect = lambda a, b: np.subtract(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.multiply.side_effect = lambda a, b: np.multiply(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.matmul.side_effect = lambda a, b: np.matmul(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.dot.side_effect = lambda a, b: np.dot(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.maximum.side_effect = lambda a, b: np.maximum(np.asarray(a), np.asarray(b)).view(_MockJaxArray)
    mock_jnp.exp.side_effect = lambda x: np.exp(np.asarray(x)).view(_MockJaxArray)
    mock_jnp.sin.side_effect = lambda x: np.sin(np.asarray(x)).view(_MockJaxArray)
    mock_jnp.sum.side_effect = lambda x, axis=None, keepdims=False: np.asarray(
        np.sum(np.asarray(x), axis=axis, keepdims=keepdims)
    ).view(_MockJaxArray)

    mock_jax = MagicMock()
    mock_jax.Array = _MockJaxArray
    mock_jax.numpy = mock_jnp
    mock_jax.nn.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockJaxArray)
    mock_jax.jit.side_effect = lambda fn, *a, **kw: fn
    mock_jax.default_backend.return_value = "cpu"
    mock_jax.devices.return_value = []
    mock_jax.local_devices.return_value = []
    mock_jax.tree_util.tree_leaves.side_effect = lambda x: (
        list(x.values()) if isinstance(x, dict) else (list(x) if isinstance(x, (list, tuple)) else [x])
    )
    for mod_name, mod_obj in (
        ("jax", mock_jax),
        ("jax.numpy", mock_jnp),
        ("jax.lax", mock_jax.lax),
        ("jax.nn", mock_jax.nn),
        ("jax.ops", mock_jax.ops),
        ("jax.scipy", mock_jax.scipy),
        ("jax.scipy.special", mock_jax.scipy.special),
        ("jax.scipy.signal", mock_jax.scipy.signal),
        ("jax.scipy.stats", mock_jax.scipy.stats),
        ("jax.scipy.linalg", mock_jax.scipy.linalg),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    for target_mod in (jax_prof, jax_eager):
        monkeypatch.setattr(target_mod, "jax", mock_jax)
        monkeypatch.setattr(target_mod, "jnp", mock_jnp)
    monkeypatch.setattr(jax_eager, "lax", mock_jax.lax)
    monkeypatch.setattr(jax_types, "jnp", mock_jnp)


def _has_working_mlx() -> bool:
    """Check whether mlx.core can be imported and loaded cleanly.

    Returns:
        bool: True if mlx.core imports without error.
    """
    import importlib

    try:
        return importlib.import_module("mlx.core") is not None
    except Exception:
        return False


def _install_mlx_and_sparse_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install NumPy-backed mock mlx and sparse modules when not installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest MonkeyPatch instance.
    """
    import importlib.util
    import sys
    from unittest.mock import MagicMock

    if not _has_working_mlx():
        import ml_switcheroo_compiler.backends.mlx.eager as mlx_eager
        import ml_switcheroo_compiler.backends.mlx.profiler as mlx_prof
        import ml_switcheroo_compiler.backends.mlx.types as mlx_types

        mock_mx = MagicMock()
        mock_mx.float32 = np.float32
        mock_mx.array.side_effect = lambda v, dtype=None: np.array(v, dtype=np.float32)
        mock_mx.zeros.side_effect = lambda s, dtype=None: np.zeros(s, dtype=np.float32)
        mock_mx.zeros_like.side_effect = lambda x: np.zeros_like(np.asarray(x))
        mock_mx.add.side_effect = lambda a, b: np.add(np.asarray(a), np.asarray(b))
        mock_mx.subtract.side_effect = lambda a, b: np.subtract(np.asarray(a), np.asarray(b))
        mock_mx.multiply.side_effect = lambda a, b: np.multiply(np.asarray(a), np.asarray(b))
        mock_mx.matmul.side_effect = lambda a, b: np.matmul(np.asarray(a), np.asarray(b))
        mock_mx.maximum.side_effect = lambda a, b: np.maximum(np.asarray(a), np.asarray(b))
        mock_mx.exp.side_effect = lambda x: np.exp(np.asarray(x))
        mock_mx.sin.side_effect = lambda x: np.sin(np.asarray(x))
        mock_mx.eval.side_effect = lambda *a: None
        mock_mx.compile.side_effect = lambda fn, *a, **kw: fn
        mock_mx.metal.is_available.return_value = False
        mock_mx.metal.get_peak_memory.return_value = 0
        mock_mx.metal.reset_peak_memory.return_value = None
        mock_mlx = MagicMock()
        mock_mlx.core = mock_mx
        for mod_name, mod_obj in (("mlx", mock_mlx), ("mlx.core", mock_mx), ("mlx.nn", mock_mlx.nn)):
            monkeypatch.setitem(sys.modules, mod_name, mod_obj)
        monkeypatch.setattr(mlx_prof, "mx", mock_mx)
        monkeypatch.setattr(mlx_types, "mx", mock_mx)
        monkeypatch.setattr(mlx_eager, "mx", mock_mx)

    if importlib.util.find_spec("sparse") is None:
        import ml_switcheroo_compiler.backends.sparse.profiler as sp_prof

        mock_sparse = MagicMock()
        mock_sparse.COO.from_numpy.side_effect = lambda arr: MagicMock(
            coords=np.zeros((np.asarray(arr).ndim, 1), dtype=np.int64),
            data=np.zeros((1,), dtype=np.float32),
            shape=np.asarray(arr).shape,
        )
        mock_sparse.GCXS.from_numpy.side_effect = lambda arr: MagicMock(
            shape=np.asarray(arr).shape,
            todense=lambda: np.asarray(arr),
        )
        monkeypatch.setitem(sys.modules, "sparse", mock_sparse)
        monkeypatch.setattr(sp_prof, "sparse", mock_sparse)


def install_profiler_backend_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install NumPy-backed mock modules for optional profiler backends when not installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest MonkeyPatch instance.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        _install_torch_stub(monkeypatch)
    if importlib.util.find_spec("jax") is None:
        _install_jax_stub(monkeypatch)
    _install_mlx_and_sparse_stubs(monkeypatch)
