"""Lightweight mock device, tensor, and neural module stubs for backend tests."""

from __future__ import annotations

import types
import typing

import numpy as np
import pytest
from ml_switcheroo_ir import EdgeList, LogicalEdge, LogicalGraph, LogicalNode, TensorShape
from ml_switcheroo_ir.graph import VersionedOutputList
from pydantic import ConfigDict

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendOperand,
    ShapeProtocol,
    coerce_shape,
)
from ml_switcheroo_compiler.core._shape_protocols import DataHolderProtocol, DTypeHolderProtocol
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.shape_system import SymNode

_CmpValT = typing.TypeVar("_CmpValT")
_SelfStubT = typing.TypeVar("_SelfStubT", bound="MockTestStub")
_SelfTensorT = typing.TypeVar("_SelfTensorT", bound="MockTensorStub")
_SelfNodeT = typing.TypeVar("_SelfNodeT", bound="MockNodeStub")
_SelfGraphT = typing.TypeVar("_SelfGraphT", bound="MockGraphStub")


class _MockTorchDevice:
    """Mock torch.device implementation."""

    def __init__(self, dev_type: BackendArg = "cpu") -> None:
        """Initialize mock device.

        Args:
            dev_type (BackendArg): Device type string or object.
        """
        self.type = str(dev_type).split(":")[0]

    def __str__(self) -> str:
        """Return string representation of device type.

        Returns:
            str: Device type string.
        """
        return self.type

    def __eq__(self, other: _CmpValT) -> bool:
        """Check device equality.

        Args:
            other (_CmpValT): Other device object.

        Returns:
            bool: True if equal.
        """
        return self.type == str(other).split(":")[0]


class _MockTorchTensor(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """NumPy-backed mock torch.Tensor."""

    def to(self, *args: BackendArg, **kwargs: BackendArg) -> _MockTorchTensor:
        """Move tensor to target device or dtype.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

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

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
        cloned = np.asarray(self).copy().view(_MockTorchTensor)
        assert isinstance(cloned, _MockTorchTensor)
        return cloned

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

    def expand(self, *sizes: BackendArg) -> _MockTorchTensor:
        """Broadcast tensor to target sizes.

        Args:
            *sizes (BackendArg): Target broadcast shape.

        Returns:
            _MockTorchTensor: Broadcasted tensor.
        """
        target = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) else sizes
        expanded = np.broadcast_to(np.asarray(self), coerce_shape(target)).copy().view(_MockTorchTensor)
        assert isinstance(expanded, _MockTorchTensor)
        return expanded


class _MockNNModule:
    """Mock torch.nn.Module base class."""

    def __init__(self) -> None:
        """Initialize mock module."""
        self.const: BackendArg = None
        self.const_0: BackendArg = None
        self.const_1: BackendArg = None
        self.const_2: BackendArg = None
        self.weight: BackendArg = None
        self.bias: BackendArg = None

    def to(self, *args: BackendArg, **kwargs: BackendArg) -> _MockNNModule:
        """Move module to device.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            _MockNNModule: Self module.
        """
        del args, kwargs
        return self

    def register_parameter(self, name: str, param: BackendArg) -> None:
        """Register parameter on module.

        Args:
            name (str): Parameter name.
            param (BackendArg): Parameter tensor.
        """
        if name == "const":
            self.const = param
        elif name == "const_0":
            self.const_0 = param
        elif name == "const_1":
            self.const_1 = param
        elif name == "const_2":
            self.const_2 = param
        elif name == "weight":
            self.weight = param
        elif name == "bias":
            self.bias = param

    def forward(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Forward pass placeholder.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Forward output.
        """
        del kwargs
        return args[0] if args else None

    def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Invoke forward method.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Forward output.
        """
        return self.forward(*args, **kwargs)


class _MockNNLinear(_MockNNModule):
    """Mock torch.nn.Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize mock linear layer.

        Args:
            in_features (int): Input feature count.
            out_features (int): Output feature count.
            bias (bool): Whether to include bias.
        """
        super().__init__()
        self.weight = np.zeros((out_features, in_features), dtype=np.float32).view(_MockTorchTensor)
        self.bias = np.zeros((out_features,), dtype=np.float32).view(_MockTorchTensor) if bias else None

    def forward(self, *args: BackendArg, **kwargs: BackendArg) -> _MockTorchTensor:
        """Compute linear projection.

        Args:
            *args (BackendArg): Positional arguments (first is input tensor).
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            _MockTorchTensor: Projected tensor.
        """
        del kwargs
        x = args[0] if args else 0.0
        res = np.matmul(np.asarray(x), np.swapaxes(np.asarray(self.weight), -1, -2))
        if self.bias is not None:
            res = res + np.asarray(self.bias)
        return np.asarray(res, dtype=np.float32).view(_MockTorchTensor)


class _MockJaxArray(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """NumPy-backed mock jax.Array."""

    def block_until_ready(self) -> _MockJaxArray:
        """Synchronize array evaluation.

        Returns:
            _MockJaxArray: Self array.
        """
        return self


class MockTestStub(types.ModuleType):
    """Typed test-double base class satisfying `ModuleType` and `OpValue`."""

    def __new__(cls: type[_SelfStubT], *args: BackendOperand, **kwargs: BackendOperand) -> _SelfStubT:
        """Allocate and initialize module namespace so subclasses need not call `super().__init__`.

        Args:
            *args (BackendOperand): Ignored positional constructor arguments.
            **kwargs (BackendOperand): Ignored keyword constructor arguments.

        Returns:
            _SelfStubT: Allocated test-double instance.
        """
        del args, kwargs
        inst = super().__new__(cls)
        types.ModuleType.__init__(inst, cls.__name__)
        return inst

    def __init__(self, name: str | None = None) -> None:
        """Initialize stub module with `name` (defaulting to the concrete class name).

        Args:
            name (str | None): Optional stub module identifier.
        """
        super().__init__(name if name is not None else type(self).__name__)

    def __dir__(self) -> list[str]:
        """Include both module instance symbols and subclass attributes in `dir(self)`.

        Returns:
            list[str]: Sorted attribute names exposed on the stub instance and its class.
        """
        return sorted(set(super().__dir__()) | set(dir(type(self))))


def make_mock_module(**attrs: BackendOperand) -> types.ModuleType:
    """Create a `types.ModuleType` test double populated with `attrs`.

    Args:
        **attrs (BackendOperand): Named attributes to attach to the module.

    Returns:
        types.ModuleType: Populated module test double.
    """
    mod = types.ModuleType("mock_module")
    mp = pytest.MonkeyPatch()
    for key, val in attrs.items():
        mp.setattr(mod, key, val, raising=False)
    return mod


class MockTensorStub(Tensor):
    """Typed `Tensor` test double allowing subclass `__init__` without invoking `Tensor.__init__`."""

    _ndim_override: int | None = None
    _size_override: int | None = None
    _has_custom_data: bool = False

    def __new__(cls: type[_SelfTensorT], *args: BackendOperand, **kwargs: BackendOperand) -> _SelfTensorT:
        """Allocate `Tensor` instance with safe defaults for test doubles.

        Args:
            *args (BackendOperand): Ignored constructor positional args.
            **kwargs (BackendOperand): Ignored constructor keyword args.

        Returns:
            _SelfTensorT: Allocated test tensor instance.
        """
        from ml_switcheroo_compiler.core.device import Device
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        del args, kwargs
        inst = super().__new__(cls)
        inst._data = 0.0
        inst._has_custom_data = False
        inst._shape = ()
        inst._dtype = DType.Float32
        inst._device = Device("cpu")
        inst._requires_grad = False
        inst._ndim_override = None
        inst._size_override = None
        inst.config = TensorConfig()
        return inst

    def __init__(self, *args: BackendOperand, **kwargs: BackendOperand) -> None:
        """No-op initializer for `MockTensorStub` subclasses.

        Args:
            *args (BackendOperand): Ignored positional args.
            **kwargs (BackendOperand): Ignored keyword args.
        """
        del args, kwargs

    @property
    def shape(self) -> tuple[int | str | SymNode, ...]:
        """Return stub tensor shape.

        Returns:
            tuple[int | str | SymNode, ...]: Shape tuple.
        """
        return self._shape

    @shape.setter
    def shape(self, value: BackendOperand) -> None:
        """Assign stub tensor shape.

        Args:
            value (BackendOperand): Shape tuple or sequence.
        """
        if isinstance(value, tuple):
            self._shape = tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in value)
        elif isinstance(value, list):
            self._shape = tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in value)
        else:
            self._shape = ()

    @property
    def dtype(self) -> DType:
        """Return stub tensor dtype.

        Returns:
            DType: Tensor dtype.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: BackendOperand) -> None:
        """Assign stub tensor dtype.

        Args:
            value (BackendOperand): DType enum or string name.
        """
        if isinstance(value, DType):
            self._dtype = value
        else:
            try:
                self._dtype = DType(str(value))
            except ValueError:
                self._dtype = DType.Float32

    @property
    def device(self) -> Device:
        """Return stub tensor device.

        Returns:
            Device: Tensor device.
        """
        return self._device

    @device.setter
    def device(self, value: BackendOperand) -> None:
        """Assign stub tensor device.

        Args:
            value (BackendOperand): Device instance or string.
        """
        self._device = value if isinstance(value, Device) else Device(str(value))

    @property
    def requires_grad(self) -> bool:
        """Return whether gradient tracking is enabled.

        Returns:
            bool: Gradient tracking flag.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Assign gradient tracking flag.

        Args:
            value (bool): Gradient tracking flag.
        """
        self._requires_grad = bool(value)

    @property
    def data(self) -> OpValue:
        """Return underlying stub data payload if explicitly assigned.

        Returns:
            OpValue: Underlying data payload.

        Raises:
            AttributeError: If no `.data` payload was assigned on this stub.
        """
        if not self._has_custom_data:
            raise AttributeError("MockTensorStub has no assigned data")
        return self._data

    @data.setter
    def data(self, value: OpValue) -> None:
        """Assign underlying stub data payload.

        Args:
            value (OpValue): Data payload.
        """
        self._data = value
        self._has_custom_data = True

    @property
    def size(self) -> int | None:
        """Return stub tensor element count or explicit override.

        Returns:
            int | None: Element count or explicit override.
        """
        if self._size_override is not None:
            return self._size_override
        return super().size

    @size.setter
    def size(self, value: int | None) -> None:
        """Assign explicit stub tensor size override.

        Args:
            value (int | None): Size override value.
        """
        self._size_override = int(value) if value is not None else None

    @property
    def ndim(self) -> int:
        """Return stub tensor rank.

        Returns:
            int: Number of dimensions.
        """
        return self._ndim_override if self._ndim_override is not None else len(self._shape)

    @ndim.setter
    def ndim(self, value: int) -> None:
        """Assign explicit stub tensor rank override.

        Args:
            value (int): Rank value.
        """
        self._ndim_override = int(value)


class MockNodeStub(LogicalNode):
    """Typed `LogicalNode` test double supporting arbitrary extra attributes."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=False)
    _in_construct: typing.ClassVar[bool] = False
    id: str = "mock"
    op_type: str = "Identity"

    def __new__(cls: type[_SelfNodeT], *args: BackendOperand, **kwargs: BackendOperand) -> _SelfNodeT:
        """Construct unvalidated `LogicalNode` test double via `model_construct`.

        Args:
            *args (BackendOperand): Optional positional `(id, op_type)` args.
            **kwargs (BackendOperand): Optional field overrides.

        Returns:
            _SelfNodeT: Constructed node stub instance.
        """
        if MockNodeStub._in_construct:
            return super().__new__(cls)
        MockNodeStub._in_construct = True
        try:
            nid = str(args[0]) if len(args) >= 1 else str(kwargs.get("id", "mock"))
            op = str(args[1]) if len(args) >= 2 else str(kwargs.get("op_type", "Identity"))
            inst = cls.model_construct(id=nid, op_type=op, outputs=VersionedOutputList([nid]))
            assert isinstance(inst, cls)
            mp = pytest.MonkeyPatch()
            if "inputs" in kwargs:
                inps_raw = kwargs["inputs"]
                inps_val = list(inps_raw) if isinstance(inps_raw, (list, tuple)) else inps_raw
                mp.setattr(inst, "inputs", inps_val, raising=False)
            if "attributes" in kwargs:
                attrs_raw = kwargs["attributes"]
                attrs_val = dict(attrs_raw) if isinstance(attrs_raw, dict) else attrs_raw
                mp.setattr(inst, "attributes", attrs_val, raising=False)
            for extra_k, extra_v in kwargs.items():
                if extra_k not in ("id", "op_type", "inputs", "attributes"):
                    if extra_k == "outputs" and isinstance(extra_v, (list, tuple)):
                        extra_v = VersionedOutputList(str(x) for x in extra_v)
                    mp.setattr(inst, extra_k, extra_v, raising=False)
            return inst
        finally:
            MockNodeStub._in_construct = False

    def __init__(self, *args: BackendOperand, **kwargs: BackendOperand) -> None:
        """No-op initializer preserving `model_construct` fields while allowing subclass overrides.

        Args:
            *args (BackendOperand): Ignored positional args.
            **kwargs (BackendOperand): Ignored keyword args.
        """
        del args, kwargs


class MockGraphStub(LogicalGraph):
    """Typed `LogicalGraph` test double supporting arbitrary extra attributes."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=False)
    _in_construct: typing.ClassVar[bool] = False
    _edges_override: tuple[LogicalEdge, ...] | None = None

    def __new__(cls: type[_SelfGraphT], *args: BackendOperand, **kwargs: BackendOperand) -> _SelfGraphT:
        """Construct unvalidated `LogicalGraph` test double via `model_construct`.

        Args:
            *args (BackendOperand): Optional positional `(name,)` args.
            **kwargs (BackendOperand): Optional field overrides.

        Returns:
            _SelfGraphT: Constructed graph stub instance.
        """
        if MockGraphStub._in_construct:
            return super().__new__(cls)
        MockGraphStub._in_construct = True
        try:
            g_name = str(args[0]) if len(args) >= 1 else str(kwargs.get("name", "mock"))
            inst = cls.model_construct(name=g_name)
            assert isinstance(inst, cls)
            mp = pytest.MonkeyPatch()
            for extra_k, extra_v in kwargs.items():
                if extra_k != "name":
                    mp.setattr(inst, extra_k, extra_v, raising=False)
            return inst
        finally:
            MockGraphStub._in_construct = False

    def __init__(self, *args: BackendOperand, **kwargs: BackendOperand) -> None:
        """No-op initializer preserving `model_construct` fields while allowing subclass overrides.

        Args:
            *args (BackendOperand): Ignored positional args.
            **kwargs (BackendOperand): Ignored keyword args.
        """
        del args, kwargs

    @property
    def edges(self) -> EdgeList:
        """Return overridden edges list if assigned, otherwise computed graph edges.

        Returns:
            EdgeList: Graph edges list.
        """
        if self._edges_override is not None:
            return EdgeList(self._edges_override)
        return super().edges

    @edges.setter
    def edges(self, value: typing.Sequence[LogicalEdge]) -> None:
        """Assign explicit edges sequence on graph stub.

        Args:
            value (typing.Sequence[LogicalEdge]): Edges sequence to attach.
        """
        self._edges_override = tuple(value)


class DummyDimSpecs(MockTestStub):
    """Dummy dimension specifications for convolution tests."""

    def __init__(self) -> None:
        """Initialize the dummy dimension specs."""
        super().__init__("DummyDimSpecs")
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


def extract_test_shape(val: BackendOperand | types.ModuleType) -> tuple[int | str | SymNode, ...]:
    """Extract `.shape` tuple from a tensor, backend array, or shape-bearing test object without array coercion.

    Args:
        val (BackendOperand | types.ModuleType): Candidate tensor, array, or shape-bearing object.

    Returns:
        tuple[int | str | SymNode, ...]: Shape tuple if `val` exposes `.shape`, otherwise `()`.
    """
    if isinstance(val, LogicalNode):
        if val.shape_metadata is None:
            return ()
        return tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in val.shape_metadata)
    if isinstance(val, TensorShape):
        return tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in val.dims)
    if type(val) is not types.ModuleType and isinstance(val, ShapeProtocol):
        raw = val.shape
        if isinstance(raw, tuple):
            return tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in raw)
        return tuple(d if isinstance(d, (int, str, SymNode)) else int(str(d)) for d in raw)
    return ()


def extract_test_dtype(val: BackendOperand | types.ModuleType) -> BackendOperand:
    """Extract `.dtype` directly from a tensor, backend array, or dtype-bearing test object.

    Args:
        val (BackendOperand | types.ModuleType): Candidate tensor, array, or dtype-bearing object.

    Returns:
        BackendOperand: Extracted `.dtype` attribute or `np.asarray(val).dtype` fallback for scalars.
    """
    if type(val) is not types.ModuleType and isinstance(val, DTypeHolderProtocol):
        return val.dtype
    return np.asarray(val).dtype


def extract_test_data(val: BackendOperand | types.ModuleType) -> BackendOperand:
    """Extract `.data` directly from a tensor or data-bearing test double without wrapping mocks.

    Args:
        val (BackendOperand | types.ModuleType): Candidate tensor or mock object.

    Returns:
        BackendOperand: Underlying `.data` payload if present, otherwise `val`.
    """
    if type(val) is not types.ModuleType and isinstance(val, DataHolderProtocol):
        return val.data
    return val
