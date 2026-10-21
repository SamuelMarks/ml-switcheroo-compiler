"""Unit tests for Jacobian transforms (jacfwd, jacrev), HVP, and auxiliary return values in JVP/VJP."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import hvp, jacfwd, jacrev, jvp, vjp
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue
GradOptions = as_tensor(getattr(importlib.import_module("ml_switcheroo_compiler.grad.jvp_vjp"), "GradOptions"))


def _make_tensor(val: EagerValue, shape: tuple[int, ...] | None = None) -> Tensor:
    """Helper to construct Tensor from array values.

    Args:
        val (EagerValue): Input data values.
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
        """Test function f_aux.

        Args:
            x: Test parameter `x`.
        """
        return x * x, "aux_info"

    (val, aux), out_tan = as_callable(jvp)(f_aux, as_callable(_make_tensor)([2.0]), as_callable(_make_tensor)([1.0]), has_aux=True)
    assert aux == "aux_info"
    assert as_tensor(getattr(val, "item"))() == 4.0
    assert as_tensor(getattr(out_tan, "item"))() == 4.0

    # 2. backend with execute_graph
    mock_backend_exec = MagicMock()
    mock_backend_exec.execute_graph.side_effect = lambda graph, inps: {out: np.array([4.0]) for out in graph.outputs}
    mock_backend_exec.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_exec):
        val_exec, out_tan_exec = as_callable(jvp)(lambda x: x * 2.0, as_callable(_make_tensor)([2.0]), as_callable(_make_tensor)([1.0]))
        assert mock_backend_exec.execute_graph.called

    # 3. backend with compile_graph
    mock_backend_comp = MagicMock(spec=["compile_graph", "asarray"])
    mock_backend_comp.compile_graph.side_effect = lambda graph: lambda inps: {out: np.array([4.0]) for out in graph.outputs}
    mock_backend_comp.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_comp):
        val_comp, out_tan_comp = as_callable(jvp)(lambda x: x * 2.0, as_callable(_make_tensor)([2.0]), as_callable(_make_tensor)([1.0]))
        assert mock_backend_comp.compile_graph.called


def test_vjp_backend_execution() -> None:
    """Verify vjp with backend execute_graph and compile_graph execution branches."""
    # 1. backend with execute_graph
    mock_backend_exec = MagicMock()
    mock_backend_exec.execute_graph.side_effect = lambda graph, inps: {out: np.array([2.0]) for out in graph.outputs}
    mock_backend_exec.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_exec):
        val_exec, vjp_fn_exec = as_callable(vjp)(lambda x: x * 2.0, as_callable(_make_tensor)([2.0]))
        as_callable(vjp_fn_exec)(as_callable(_make_tensor)([1.0]))
        assert mock_backend_exec.execute_graph.called

    # 2. backend with compile_graph
    mock_backend_comp = MagicMock(spec=["compile_graph", "asarray"])
    mock_backend_comp.compile_graph.side_effect = lambda graph: lambda inps: {out: np.array([2.0]) for out in graph.outputs}
    mock_backend_comp.asarray = np.asarray

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.get_active_backend", return_value=mock_backend_comp):
        val_comp, vjp_fn_comp = as_callable(vjp)(lambda x: x * 2.0, as_callable(_make_tensor)([2.0]))
        as_callable(vjp_fn_comp)(as_callable(_make_tensor)([1.0]))
        assert mock_backend_comp.compile_graph.called


def test_hvp_has_aux_and_projected_tangents() -> None:
    """Verify hvp with has_aux=True and projected_tangents branches."""

    # 1. has_aux=True
    def f_aux(x: Tensor) -> tuple[Tensor, str]:
        """Test function f_aux.

        Args:
            x: Test parameter `x`.
        """
        return x * x * x, "aux_meta"

    (val, aux), out_tan = as_callable(hvp)(f_aux, as_callable(_make_tensor)([2.0]), as_callable(_make_tensor)([1.0]), has_aux=True)
    assert aux == "aux_meta"
    assert as_tensor(getattr(val, "item"))() == 8.0
    assert as_tensor(getattr(out_tan, "item"))() == 12.0

    # 2. projected_tangents as a single Tensor
    def f_cube(x: Tensor) -> Tensor:
        """Test function f_cube.

        Args:
            x: Test parameter `x`.
        """
        return x * x * x

    val_pt, out_tan_pt = as_callable(hvp)(
        f_cube,
        _make_tensor([2.0]),
        _make_tensor([1.0]),
        projected_tangents=_make_tensor([1.0]),
    )
    assert as_tensor(getattr(val_pt, "item"))() == 8.0
    assert out_tan_pt is not None

    # 3. projected_tangents as a tuple of Tensors
    val_pt2, out_tan_pt2 = as_callable(hvp)(
        f_cube,
        _make_tensor([2.0]),
        _make_tensor([1.0]),
        projected_tangents=(_make_tensor([1.0]),),
    )
    assert as_tensor(getattr(val_pt2, "item"))() == 8.0
    assert out_tan_pt2 is not None


def test_jacfwd_and_jacrev_shapes_and_aux() -> None:
    """Verify jacfwd and jacrev for scalar outputs, auxiliary returns, and multi-outputs."""

    # 1. jacfwd with scalar output (ndim == 0)
    def f_scalar(x: Tensor) -> Tensor:
        """Test function f_scalar.

        Args:
            x: Test parameter `x`.
        """
        return x[0]

    j_fwd = as_callable(jacfwd)(f_scalar)(as_callable(_make_tensor)([2.0, 3.0]))
    assert as_tensor(getattr(j_fwd, "shape")) == (1, 2)

    # 2. jacfwd with has_aux=True
    def f_aux_fwd(x: Tensor) -> tuple[Tensor, str]:
        """Test function f_aux_fwd.

        Args:
            x: Test parameter `x`.
        """
        return x * x, "aux_info_fwd"

    opts_fwd = GradOptions(has_aux=True)
    j_fwd_aux = as_callable(jacfwd)(f_aux_fwd, options=opts_fwd)(as_callable(_make_tensor)([2.0, 3.0]))
    assert as_tensor(getattr(j_fwd_aux, "shape")) == (2, 2)

    # 3. jacrev with has_aux=True
    def f_aux_rev(x: Tensor) -> tuple[Tensor, str]:
        """Test function f_aux_rev.

        Args:
            x: Test parameter `x`.
        """
        return x * x, "aux_info"

    opts_rev = GradOptions(has_aux=True)
    j_rev_aux = as_callable(jacrev)(f_aux_rev, options=opts_rev)(as_callable(_make_tensor)([2.0, 3.0]))
    assert as_tensor(getattr(j_rev_aux, "shape")) == (2, 2)

    # 4. jacrev with multi-output
    def f_multi(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function f_multi.

        Args:
            x: Test parameter `x`.
        """
        return x * 2.0, x * 3.0

    j_multi = as_callable(jacrev)(f_multi)(as_callable(_make_tensor)([2.0, 3.0]))
    assert as_tensor(getattr(j_multi, "shape")) == (4, 2)

    # 5. jacrev with scalar output
    j_rev_scalar = as_callable(jacrev)(f_scalar)(as_callable(_make_tensor)([2.0, 3.0]))
    assert as_tensor(getattr(j_rev_scalar, "shape")) == (2,)
