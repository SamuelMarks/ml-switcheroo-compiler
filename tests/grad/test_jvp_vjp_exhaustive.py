"""Exhaustive tests for jvp_vjp module edge cases and branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import (
    GradOptions,
    hvp,
    jacfwd,
    jacrev,
    jvp,
    vjp,
)


def _make_tensor(val: object, shape: tuple[int, ...] | None = None) -> Tensor:
    """Helper to construct Tensor from array values.

    Args:
        val (object): Input data values.
        shape (tuple[int, ...] | None): Tensor dimensions.

    Returns:
        Tensor: Constructed tensor instance.
    """
    arr = np.array(val, dtype=np.float32)
    if shape is None:
        shape = arr.shape
    return Tensor(arr, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_jvp_has_aux_and_backend_execution() -> None:
    """Verify jvp with has_aux=True and backend execute_graph/compile_graph branches."""

    # 1. has_aux=True
    def f_aux(x: Tensor) -> tuple[Tensor, str]:
        return x * x, "aux_info"

    (val, aux), out_tan = jvp(f_aux, _make_tensor([2.0]), _make_tensor([1.0]), has_aux=True)
    assert aux == "aux_info"
    assert val.item() == 4.0
    assert out_tan.item() == 4.0

    # 2. backend with execute_graph
    mock_backend_exec = MagicMock()
    mock_backend_exec.execute_graph.side_effect = lambda graph, inps: {out: np.array([4.0]) for out in graph.outputs}
    mock_backend_exec.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_exec):
        val_exec, out_tan_exec = jvp(lambda x: x * 2.0, _make_tensor([2.0]), _make_tensor([1.0]))
        assert mock_backend_exec.execute_graph.called

    # 3. backend with compile_graph
    mock_backend_comp = MagicMock(spec=["compile_graph", "asarray"])
    mock_backend_comp.compile_graph.side_effect = lambda graph: lambda inps: {out: np.array([4.0]) for out in graph.outputs}
    mock_backend_comp.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_comp):
        val_comp, out_tan_comp = jvp(lambda x: x * 2.0, _make_tensor([2.0]), _make_tensor([1.0]))
        assert mock_backend_comp.compile_graph.called


def test_vjp_backend_execution() -> None:
    """Verify vjp with backend execute_graph and compile_graph execution branches."""
    # 1. backend with execute_graph
    mock_backend_exec = MagicMock()
    mock_backend_exec.execute_graph.side_effect = lambda graph, inps: {out: np.array([2.0]) for out in graph.outputs}
    mock_backend_exec.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_exec):
        val_exec, vjp_fn_exec = vjp(lambda x: x * 2.0, _make_tensor([2.0]))
        grads_exec = vjp_fn_exec(_make_tensor([1.0]))
        assert mock_backend_exec.execute_graph.called

    # 2. backend with compile_graph
    mock_backend_comp = MagicMock(spec=["compile_graph", "asarray"])
    mock_backend_comp.compile_graph.side_effect = lambda graph: lambda inps: {out: np.array([2.0]) for out in graph.outputs}
    mock_backend_comp.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_comp):
        val_comp, vjp_fn_comp = vjp(lambda x: x * 2.0, _make_tensor([2.0]))
        grads_comp = vjp_fn_comp(_make_tensor([1.0]))
        assert mock_backend_comp.compile_graph.called


def test_hvp_has_aux_and_projected_tangents() -> None:
    """Verify hvp with has_aux=True and projected_tangents branches."""

    # 1. has_aux=True
    def f_aux(x: Tensor) -> tuple[Tensor, str]:
        return x * x * x, "aux_meta"

    (val, aux), out_tan = hvp(f_aux, _make_tensor([2.0]), _make_tensor([1.0]), has_aux=True)
    assert aux == "aux_meta"
    assert val.item() == 8.0
    assert out_tan.item() == 12.0

    # 2. projected_tangents as a single Tensor
    def f_cube(x: Tensor) -> Tensor:
        return x * x * x

    val_pt, out_tan_pt = hvp(
        f_cube,
        _make_tensor([2.0]),
        _make_tensor([1.0]),
        projected_tangents=_make_tensor([1.0]),
    )
    assert val_pt.item() == 8.0
    assert out_tan_pt is not None

    # 3. projected_tangents as a tuple of Tensors
    val_pt2, out_tan_pt2 = hvp(
        f_cube,
        _make_tensor([2.0]),
        _make_tensor([1.0]),
        projected_tangents=(_make_tensor([1.0]),),
    )
    assert val_pt2.item() == 8.0
    assert out_tan_pt2 is not None


def test_jacfwd_and_jacrev_shapes_and_aux() -> None:
    """Verify jacfwd and jacrev for scalar outputs, auxiliary returns, and multi-outputs."""

    # 1. jacfwd with scalar output (ndim == 0)
    def f_scalar(x: Tensor) -> Tensor:
        return x[0]

    j_fwd = jacfwd(f_scalar)(_make_tensor([2.0, 3.0]))
    assert j_fwd.shape == (1, 2)

    # 2. jacfwd with has_aux=True
    def f_aux_fwd(x: Tensor) -> tuple[Tensor, str]:
        return x * x, "aux_info_fwd"

    opts_fwd = GradOptions(has_aux=True)
    j_fwd_aux = jacfwd(f_aux_fwd, options=opts_fwd)(_make_tensor([2.0, 3.0]))
    assert j_fwd_aux.shape == (2, 2)

    # 3. jacrev with has_aux=True
    def f_aux_rev(x: Tensor) -> tuple[Tensor, str]:
        return x * x, "aux_info"

    opts_rev = GradOptions(has_aux=True)
    j_rev_aux = jacrev(f_aux_rev, options=opts_rev)(_make_tensor([2.0, 3.0]))
    assert j_rev_aux.shape == (2, 2)

    # 4. jacrev with multi-output
    def f_multi(x: Tensor) -> tuple[Tensor, Tensor]:
        return x * 2.0, x * 3.0

    j_multi = jacrev(f_multi)(_make_tensor([2.0, 3.0]))
    assert j_multi.shape == (4, 2)

    # 5. jacrev with scalar output
    j_rev_scalar = jacrev(f_scalar)(_make_tensor([2.0, 3.0]))
    assert j_rev_scalar.shape == (2,)
