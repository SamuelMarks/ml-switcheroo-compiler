"""Tests for test_grad_utils."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.grad.utils import _get_inputs_dict, _to_original_type
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_get_inputs_dict_missing_input(mocker: EagerTensorProtocol) -> None:
    """Verify the behavior and output invariants of test get inputs dict missing input.

    Args:
        mocker: Test parameter `mocker`.
    """

    class DummyGraph:
        """Test class DummyGraph."""

        inputs = ["missing_inp"]
        nodes = ensure_tensor_dict({})

    with pytest.raises(ValueError, match="Missing input value for node 'missing_inp'"):
        ensure_callable(_get_inputs_dict)(ensure_callable(DummyGraph)())


def test_to_original_type_dtypes(mocker: EagerTensorProtocol) -> None:
    """Verify the behavior and output invariants of test to original type dtypes.

    Args:
        mocker: Test parameter `mocker`.
    """
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig

    orig = Tensor(None, TensorConfig((), DType.Float32, None))

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, val: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                val: Test parameter `val`.
            """
            return val

    ensure_callable(mocker.patch)("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=ensure_callable(DummyBackend)())

    val_int = np.array(1, dtype=np.int32)
    res_int = _to_original_type(val_int, orig)
    assert ensure_eager_tensor(ensure_eager_tensor(res_int).dtype) == DType.Int32

    val_bool = np.array(True, dtype=bool)
    res_bool = _to_original_type(val_bool, orig)
    assert ensure_eager_tensor(ensure_eager_tensor(res_bool).dtype) == DType.Bool


def test_find_wrt_tensors(mocker: EagerTensorProtocol) -> None:
    """Verify the behavior and output invariants of test find wrt tensors.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.utils import _find_wrt_tensors

    class DummyData:
        """Test class DummyData."""

        id = "n"

    class DummyGraph:
        """Test class DummyGraph."""

        nodes = ensure_eager_tensor({"n": {}})

    class DummyTensor1(Tensor):
        """Test class DummyTensor1."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad."""
            return True

        @property
        def trainable(self) -> EagerValue:
            """Test function trainable."""
            return False

    class DummyTensor2(Tensor):
        """Test class DummyTensor2."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad."""
            return False

        @property
        def trainable(self) -> EagerValue:
            """Test function trainable."""
            return True

    t = DummyTensor1(None, TensorConfig((), DType.Float32, None))
    ensure_mutable(t)._data = ensure_eager_tensor(DummyData())
    t2 = DummyTensor2(None, TensorConfig((), DType.Float32, None))
    ensure_mutable(t2)._data = ensure_eager_tensor(DummyData())

    class DummyTensor3(Tensor):
        """Test class DummyTensor3."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad."""
            return True

    class DummyDataNotInGraph:
        """Test class DummyDataNotInGraph."""

        id = "not_in_graph"

    t3 = DummyTensor3(None, TensorConfig((), DType.Float32, None))
    ensure_mutable(t3)._data = ensure_eager_tensor(DummyDataNotInGraph())

    wrt_tensors, wrt_ids = ensure_callable(_find_wrt_tensors)(ensure_callable(DummyGraph)())
    assert id(t) in [id(x) for x in wrt_tensors]
    assert id(t2) in [id(x) for x in wrt_tensors]
    assert id(t3) not in [id(x) for x in wrt_tensors]
    assert "n" in wrt_ids


def test_get_concrete_val_variants() -> None:
    """Test _get_concrete_val with non-proxy and proxy objects."""
    from ml_switcheroo_compiler.grad.utils import _get_concrete_val

    class NonProxy:
        """Test class NonProxy."""

        _data = 42

    assert ensure_callable(_get_concrete_val)(ensure_callable(NonProxy)()) == 42


def test_get_inputs_dict_success(mocker: EagerTensorProtocol) -> None:
    """Test _get_inputs_dict full branch coverage including successful inputs lookup.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, val: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                val: Test parameter `val`.
            """
            return val

    ensure_callable(mocker.patch)("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=ensure_callable(DummyBackend)())

    class NodeData:
        """Test class NodeData."""

        def __init__(self, id: EagerValue, concrete_value: EagerValue = None) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
                concrete_value: Test parameter `concrete_value`.
            """
            self.id = id
            self.concrete_value = concrete_value

    class DummyGraphSuccess:
        """Test class DummyGraphSuccess."""

        inputs = ["in_valid"]
        nodes = ensure_eager_tensor({"in_valid": {}, "no_val_node": {}})

    t_valid = Tensor(None, TensorConfig((), DType.Float32, None))
    ensure_mutable(t_valid)._data = ensure_eager_tensor(NodeData("in_valid", concrete_value=123.0))

    t_no_val = Tensor(None, TensorConfig((), DType.Float32, None))
    proxy = ensure_callable(ProxyTensor)(None, ())
    ensure_mutable(proxy).id = ensure_eager_tensor("no_val_node")
    ensure_mutable(proxy).concrete_value = ensure_eager_tensor(None)
    t_no_val._data = proxy

    t_not_in_graph = Tensor(None, TensorConfig((), DType.Float32, None))
    ensure_mutable(t_not_in_graph)._data = ensure_eager_tensor(NodeData("orphan_node", concrete_value=999.0))

    Tensor(1.0, TensorConfig((), DType.Float32, None))

    res = ensure_callable(_get_inputs_dict)(ensure_callable(DummyGraphSuccess)())
    assert "in_valid" in res
    assert res["in_valid"] == 123.0
    assert "no_val_node" not in res
    assert "orphan_node" not in res

    class GraphNoInputs:
        """Test class GraphNoInputs."""

        nodes = ensure_tensor_dict({})

    res_no_inputs = ensure_callable(_get_inputs_dict)(ensure_callable(GraphNoInputs)())
    assert isinstance(ensure_eager_tensor(res_no_inputs), dict)
