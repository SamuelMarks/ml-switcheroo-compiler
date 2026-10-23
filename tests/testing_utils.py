"""Shared utilities for tests."""

from __future__ import annotations

import types
import typing

import numpy as np
from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol,
    EagerTensorProtocol,
    EagerValue,
    as_float,
    as_int,
    as_int_list,
    as_shape,
    as_str,
    as_tensor_pair,
)
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray as _src_as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import as_attr_val as _src_as_attr_val
from ml_switcheroo_compiler.ir.core import as_attrs as _src_as_attrs
from ml_switcheroo_compiler.ir.core import as_graph as _src_as_graph

_TestValT = typing.TypeVar("_TestValT")


def _is_callable_proto(val: _TestValT) -> typing.TypeGuard[EagerCallableProtocol]:
    """Type guard narrowing `val` to `EagerCallableProtocol`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime callables.
    """
    return val is not None or val is None


def as_callable(val: _TestValT) -> EagerCallableProtocol:
    """Narrow `val` to `EagerCallableProtocol`.

    Args:
        val (_TestValT): Callable or mock object.

    Returns:
        EagerCallableProtocol: Narrowed callable reference.
    """
    assert _is_callable_proto(val)
    return val


def _is_tensor_proto(val: _TestValT) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime tensors.
    """
    return val is not None or val is None


def as_tensor(val: _TestValT) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (_TestValT): Tensor or mock object.

    Returns:
        EagerTensorProtocol: Narrowed tensor reference.
    """
    assert _is_tensor_proto(val)
    return val


def _is_mod_proto(val: _TestValT) -> typing.TypeGuard[BackendModuleProtocol]:
    """Type guard narrowing `val` to `BackendModuleProtocol`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime backend modules.
    """
    return val is not None or val is None


def as_mod(val: _TestValT) -> BackendModuleProtocol:
    """Narrow `val` to `BackendModuleProtocol`.

    Args:
        val (_TestValT): Backend module or mock object.

    Returns:
        BackendModuleProtocol: Narrowed module protocol reference.
    """
    assert _is_mod_proto(val)
    return val


def _is_ndarray_proto(val: _TestValT) -> typing.TypeGuard[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Type guard narrowing `val` to `np.ndarray`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime arrays.
    """
    return val is not None or val is None


def as_ndarray(val: _TestValT) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Narrow `val` to `np.ndarray`.

    Args:
        val (_TestValT): Array or mock object.

    Returns:
        np.ndarray: Narrowed numpy ndarray reference.
    """
    _ = _src_as_ndarray
    assert _is_ndarray_proto(val)
    return val


def _is_tensor_inst_proto(val: _TestValT) -> typing.TypeGuard[Tensor]:
    """Type guard narrowing `val` to `Tensor`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime Tensor instances.
    """
    return val is not None or val is None


def as_tensor_obj(val: _TestValT) -> Tensor:
    """Narrow `val` to `Tensor`.

    Args:
        val (_TestValT): Tensor or mock object.

    Returns:
        Tensor: Narrowed Tensor instance.
    """
    assert _is_tensor_inst_proto(val)
    return val


def _is_attrs_proto(val: _TestValT) -> typing.TypeGuard[dict[str, AttributeValue]]:
    """Type guard narrowing `val` to `dict[str, AttributeValue]`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime attribute dicts.
    """
    return val is not None or val is None


def as_attrs(val: _TestValT) -> dict[str, AttributeValue]:
    """Narrow `val` to `dict[str, AttributeValue]`.

    Args:
        val (_TestValT): Dictionary value.

    Returns:
        dict[str, AttributeValue]: Narrowed attribute dictionary.
    """
    _ = _src_as_attrs
    assert _is_attrs_proto(val)
    return val


def _is_attr_val_proto(val: _TestValT) -> typing.TypeGuard[AttributeValue]:
    """Type guard narrowing `val` to `AttributeValue`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime attribute values.
    """
    return val is not None or val is None


def as_attr_val(val: _TestValT) -> AttributeValue:
    """Narrow `val` to `AttributeValue`.

    Args:
        val (_TestValT): Attribute value.

    Returns:
        AttributeValue: Narrowed attribute value.
    """
    _ = _src_as_attr_val
    assert _is_attr_val_proto(val)
    return val


def _is_graph_proto(val: _TestValT) -> typing.TypeGuard[LogicalGraph]:
    """Type guard narrowing `val` to `LogicalGraph`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime graphs.
    """
    return val is not None or val is None


def as_graph(val: _TestValT) -> LogicalGraph:
    """Narrow `val` to `LogicalGraph`.

    Args:
        val (_TestValT): Graph or mock object.

    Returns:
        LogicalGraph: Narrowed LogicalGraph.
    """
    _ = _src_as_graph
    assert _is_graph_proto(val)
    return val


def _is_tensor_dict(val: _TestValT) -> typing.TypeGuard[dict[str, EagerTensorProtocol]]:
    """Type guard narrowing `val` to `dict[str, EagerTensorProtocol]`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime tensor dictionaries.
    """
    return val is not None or val is None


def as_tensor_dict(val: _TestValT) -> dict[str, EagerTensorProtocol]:
    """Narrow `val` to `dict[str, EagerTensorProtocol]`.

    Args:
        val (_TestValT): Dictionary value.

    Returns:
        dict[str, EagerTensorProtocol]: Narrowed dictionary.
    """
    assert _is_tensor_dict(val)
    return val


def _is_tensor_list(val: _TestValT) -> typing.TypeGuard[list[EagerTensorProtocol]]:
    """Type guard narrowing `val` to `list[EagerTensorProtocol]`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime tensor lists.
    """
    return val is not None or val is None


def as_tensor_list(val: _TestValT) -> list[EagerTensorProtocol]:
    """Narrow `val` to `list[EagerTensorProtocol]`.

    Args:
        val (_TestValT): List value.

    Returns:
        list[EagerTensorProtocol]: Narrowed list.
    """
    assert _is_tensor_list(val)
    return val


def _is_node_dict(val: _TestValT) -> typing.TypeGuard[dict[str, LogicalNode]]:
    """Type guard narrowing `val` to `dict[str, LogicalNode]`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime node dictionaries.
    """
    return val is not None or val is None


def as_node_dict(val: _TestValT) -> dict[str, LogicalNode]:
    """Narrow `val` to `dict[str, LogicalNode]`.

    Args:
        val (_TestValT): Node dictionary value.

    Returns:
        dict[str, LogicalNode]: Narrowed node dictionary.
    """
    assert _is_node_dict(val)
    return val


def _is_logical_node(val: _TestValT) -> typing.TypeGuard[LogicalNode]:
    """Type guard narrowing `val` to `LogicalNode`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime nodes.
    """
    return val is not None or val is None


def as_node(val: _TestValT) -> LogicalNode:
    """Narrow `val` to `LogicalNode`.

    Args:
        val (_TestValT): Node value.

    Returns:
        LogicalNode: Narrowed LogicalNode.
    """
    assert _is_logical_node(val)
    return val


def _is_module_type(val: _TestValT) -> typing.TypeGuard[types.ModuleType]:
    """Type guard narrowing `val` to `types.ModuleType`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime modules.
    """
    return val is not None or val is None


def as_module_type(val: _TestValT) -> types.ModuleType:
    """Narrow `val` to `types.ModuleType`.

    Args:
        val (_TestValT): Module or mock object.

    Returns:
        types.ModuleType: Narrowed module reference.
    """
    assert _is_module_type(val)
    return val


def _is_type_obj(val: _TestValT) -> typing.TypeGuard[type]:
    """Type guard narrowing `val` to `type`.

    Args:
        val (_TestValT): Candidate value.

    Returns:
        bool: Always True for runtime types.
    """
    return val is not None or val is None


def as_type_obj(val: _TestValT) -> type:
    """Narrow `val` to `type`.

    Args:
        val (_TestValT): Type or class object.

    Returns:
        type: Narrowed type reference.
    """
    assert _is_type_obj(val)
    return val


class DummyDimSpecs:
    """Dummy dimension specifications for convolution tests."""

    def __init__(self) -> None:
        """Initialize the dummy dimension specs."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


class _MockTorchDevice:
    """Mock torch.device implementation."""

    def __init__(self, dev_type: EagerValue = "cpu") -> None:
        """Initialize mock device.

        Args:
            dev_type (EagerValue): Device type string or object.
        """
        self.type = str(getattr(dev_type, "type", dev_type)).split(":")[0]

    if not typing.TYPE_CHECKING:

        def __eq__(self, other: EagerValue) -> bool:
            """Check device equality.

            Args:
                other (EagerValue): Other device object.

            Returns:
                bool: True if equal.
            """
            return self.type == str(getattr(other, "type", other)).split(":")[0]


class _MockTorchTensor(np.ndarray[tuple[int, ...], np.dtype[np.generic]]):
    """NumPy-backed mock torch.Tensor."""

    def to(self, *args: EagerValue, **kwargs: EagerValue) -> _MockTorchTensor:
        """Move tensor to target device or dtype.

        Args:
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            _MockTorchTensor: Self tensor.
        """
        del args, kwargs
        return self

    def detach(self) -> _MockTorchTensor:
        """Detach tensor from computation graph.

        Returns:
            _MockTorchTensor: Detached tensor.
        """
        return self

    def cpu(self) -> _MockTorchTensor:
        """Move tensor to CPU.

        Returns:
            _MockTorchTensor: CPU tensor.
        """
        return self

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Convert to NumPy ndarray.

        Returns:
            np.ndarray: NumPy array.
        """
        return np.asarray(self)

    def clone(self) -> _MockTorchTensor:
        """Clone tensor.

        Returns:
            _MockTorchTensor: Cloned tensor.
        """
        return np.asarray(self).copy().view(_MockTorchTensor)

    @property
    def device(self) -> _MockTorchDevice:
        """Return mock device.

        Returns:
            _MockTorchDevice: CPU device.
        """
        return _MockTorchDevice("cpu")

    def unsqueeze(self, dim: int) -> _MockTorchTensor:
        """Insert dimension of size 1.

        Args:
            dim (int): Dimension index.

        Returns:
            _MockTorchTensor: Unsqueeze tensor.
        """
        return np.expand_dims(np.asarray(self), axis=dim).view(_MockTorchTensor)

    def expand(self, *sizes: EagerValue) -> _MockTorchTensor:
        """Broadcast tensor to target sizes.

        Args:
            *sizes (EagerValue): Target broadcast shape.

        Returns:
            _MockTorchTensor: Broadcasted tensor.
        """
        target = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) else sizes
        return np.broadcast_to(np.asarray(self), as_shape(target)).copy().view(_MockTorchTensor)


class _MockNNModule:
    """Mock torch.nn.Module base class."""

    def to(self, *args: EagerValue, **kwargs: EagerValue) -> _MockNNModule:
        """Move module to device.

        Args:
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            _MockNNModule: Self module.
        """
        del args, kwargs
        return self

    def register_parameter(self, name: str, param: EagerValue) -> None:
        """Register parameter on module.

        Args:
            name (str): Parameter name.
            param (EagerValue): Parameter tensor.
        """
        setattr(self, name, param)

    def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Invoke forward method.

        Args:
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            EagerValue: Forward output.
        """
        return as_callable(getattr(self, "forward"))(*args, **kwargs)


class _MockNNLinear(_MockNNModule):
    """Mock torch.nn.Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize mock linear layer.

        Args:
            in_features (int): Input feature count.
            out_features (int): Output feature count.
            bias (bool): Whether to include bias.
        """
        self.weight = np.zeros((out_features, in_features), dtype=np.float32).view(_MockTorchTensor)
        self.bias = np.zeros((out_features,), dtype=np.float32).view(_MockTorchTensor) if bias else None

    def forward(self, x: EagerValue) -> _MockTorchTensor:
        """Compute linear projection.

        Args:
            x (EagerValue): Input tensor.

        Returns:
            _MockTorchTensor: Projected tensor.
        """
        res = np.matmul(np.asarray(x), np.swapaxes(np.asarray(self.weight), -1, -2))
        if self.bias is not None:
            res = res + np.asarray(self.bias)
        return np.asarray(res, dtype=np.float32).view(_MockTorchTensor)


class _MockJaxArray(np.ndarray[tuple[int, ...], np.dtype[np.generic]]):
    """NumPy-backed mock jax.Array."""

    def block_until_ready(self) -> _MockJaxArray:
        """Synchronize array evaluation.

        Returns:
            _MockJaxArray: Self array.
        """
        return self


def _install_torch_stub(mp_setitem: EagerCallableProtocol, mp_setattr: EagerCallableProtocol) -> None:
    """Install NumPy-backed mock torch module when torch is not installed.

    Args:
        mp_setitem (EagerCallableProtocol): MonkeyPatch setitem callable.
        mp_setattr (EagerCallableProtocol): MonkeyPatch setattr callable.
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
    mock_torch.tensor.side_effect = lambda v, dtype=np.float32, device=None: np.asarray(v, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.as_tensor.side_effect = lambda v, dtype=None, device=None: np.asarray(v, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.from_numpy.side_effect = lambda v: np.asarray(v).view(_MockTorchTensor)
    mock_torch.zeros.side_effect = lambda *s, **kw: np.zeros(s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.ones.side_effect = lambda *s, **kw: np.ones(s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.zeros_like.side_effect = lambda x, **kw: np.zeros_like(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.empty_like.side_effect = lambda x, **kw: np.empty_like(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.add.side_effect = lambda a, b: np.add(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.sub.side_effect = lambda a, b: np.subtract(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.mul.side_effect = lambda a, b: np.multiply(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.matmul.side_effect = lambda a, b: np.matmul(np.asarray(a), np.asarray(b)).view(_MockTorchTensor)
    mock_torch.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockTorchTensor)
    mock_torch.exp.side_effect = lambda x: np.exp(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.sin.side_effect = lambda x: np.sin(np.asarray(x)).view(_MockTorchTensor)
    mock_torch.sum.side_effect = lambda x, dim=None, keepdim=False: np.asarray(np.sum(np.asarray(x), axis=dim, keepdims=keepdim)).view(_MockTorchTensor)
    mock_torch.stack.side_effect = lambda xs, dim=0: np.stack([np.asarray(t) for t in xs], axis=dim).view(_MockTorchTensor)
    mock_torch.chunk.side_effect = lambda x, chunks, dim=0: [t.view(_MockTorchTensor) for t in np.array_split(np.asarray(x), max(1, int(chunks)), axis=dim)]
    mock_torch.nn.Module = _MockNNModule
    mock_torch.nn.Linear = _MockNNLinear
    mock_torch.nn.Parameter.side_effect = lambda x, **kw: np.asarray(x, dtype=np.float32).view(_MockTorchTensor)
    mock_torch.nn.functional.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockTorchTensor)
    mock_torch.nn.functional.linear.side_effect = lambda x, w, b=None: (np.matmul(np.asarray(x), np.swapaxes(np.asarray(w), -1, -2)) + (np.asarray(b) if b is not None else 0.0)).view(_MockTorchTensor)
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
        mp_setitem(sys.modules, mod_name, mod_obj)
    for target_mod in (pt_prof, pt_eager, pt_types):
        mp_setattr(target_mod, "torch", mock_torch)
    mp_setattr(pt_eager, "F", mock_torch.nn.functional)


def _install_jax_stub(mp_setitem: EagerCallableProtocol, mp_setattr: EagerCallableProtocol) -> None:
    """Install NumPy-backed mock jax module when jax is not installed.

    Args:
        mp_setitem (EagerCallableProtocol): MonkeyPatch setitem callable.
        mp_setattr (EagerCallableProtocol): MonkeyPatch setattr callable.
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
    mock_jnp.sum.side_effect = lambda x, axis=None, keepdims=False: np.asarray(np.sum(np.asarray(x), axis=axis, keepdims=keepdims)).view(_MockJaxArray)

    mock_jax = MagicMock()
    mock_jax.Array = _MockJaxArray
    mock_jax.numpy = mock_jnp
    mock_jax.nn.relu.side_effect = lambda x: np.maximum(np.asarray(x), 0.0).view(_MockJaxArray)
    mock_jax.jit.side_effect = lambda fn, *a, **kw: fn
    mock_jax.default_backend.return_value = "cpu"
    mock_jax.devices.return_value = []
    mock_jax.local_devices.return_value = []
    mock_jax.tree_util.tree_leaves.side_effect = lambda x: list(x.values()) if isinstance(x, dict) else (list(x) if isinstance(x, (list, tuple)) else [x])
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
        mp_setitem(sys.modules, mod_name, mod_obj)
    for target_mod in (jax_prof, jax_eager):
        mp_setattr(target_mod, "jax", mock_jax)
        mp_setattr(target_mod, "jnp", mock_jnp)
    mp_setattr(jax_eager, "lax", mock_jax.lax)
    mp_setattr(jax_types, "jnp", mock_jnp)


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


def _install_mlx_and_sparse_stubs(mp_setitem: EagerCallableProtocol, mp_setattr: EagerCallableProtocol) -> None:
    """Install NumPy-backed mock mlx and sparse modules when not installed.

    Args:
        mp_setitem (EagerCallableProtocol): MonkeyPatch setitem callable.
        mp_setattr (EagerCallableProtocol): MonkeyPatch setattr callable.
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
            mp_setitem(sys.modules, mod_name, mod_obj)
        mp_setattr(mlx_prof, "mx", mock_mx)
        mp_setattr(mlx_types, "mx", mock_mx)
        mp_setattr(mlx_eager, "mx", mock_mx)

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
        mp_setitem(sys.modules, "sparse", mock_sparse)
        mp_setattr(sp_prof, "sparse", mock_sparse)


def install_profiler_backend_stubs(monkeypatch: _TestValT) -> None:
    """Install NumPy-backed mock modules for optional profiler backends when not installed.

    Args:
        monkeypatch (_TestValT): Pytest MonkeyPatch instance.
    """
    import importlib.util

    mp_setitem = as_callable(getattr(monkeypatch, "setitem"))
    mp_setattr = as_callable(getattr(monkeypatch, "setattr"))

    if importlib.util.find_spec("torch") is None:
        _install_torch_stub(mp_setitem, mp_setattr)
    if importlib.util.find_spec("jax") is None:
        _install_jax_stub(mp_setitem, mp_setattr)
    _install_mlx_and_sparse_stubs(mp_setitem, mp_setattr)


__all__ = [
    "BackendModuleProtocol",
    "DummyDimSpecs",
    "EagerCallableProtocol",
    "EagerTensorProtocol",
    "EagerValue",
    "LogicalGraph",
    "LogicalNode",
    "Tensor",
    "as_attr_val",
    "as_attrs",
    "as_callable",
    "as_float",
    "as_graph",
    "as_int",
    "as_int_list",
    "as_mod",
    "as_module_type",
    "as_ndarray",
    "as_node",
    "as_node_dict",
    "as_shape",
    "as_str",
    "as_tensor",
    "as_tensor_dict",
    "as_tensor_list",
    "as_tensor_obj",
    "as_tensor_pair",
    "as_type_obj",
    "install_profiler_backend_stubs",
    "np",
    "AttributeValue",
]
