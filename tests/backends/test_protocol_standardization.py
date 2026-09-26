"""Unit tests verifying adherence to standardized compiler eager and autodiff protocols."""

from typing import Union

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.grad.utils import GradTensorProtocol
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ops.eager_evaluator import (
    BackendExecuteOpStrategy,
    CustomEagerEvalStrategy,
    EvaluationContext,
    EvaluationStrategy,
)


class MockEagerTensor:
    """Mock tensor satisfying EagerTensorProtocol."""

    def __init__(self, shape: tuple[int, ...] = (2, 2)) -> None:
        """Initialize MockEagerTensor."""
        self._shape = shape
        self._dtype = "float32"

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self._shape

    @property
    def dtype(self) -> object:
        """Get dtype."""
        return self._dtype

    def __len__(self) -> int:
        """Length of leading dim."""
        return self._shape[0]

    def __getitem__(self, key: object) -> "MockEagerTensor":
        """Get item."""
        return self

    def __setitem__(self, key: object, value: object) -> None:
        """Set item."""
        pass

    def __neg__(self) -> "MockEagerTensor":
        """Negation."""
        return self

    def __add__(self, other: EagerValue) -> "MockEagerTensor":
        """Addition."""
        return self

    def __sub__(self, other: EagerValue) -> "MockEagerTensor":
        """Subtraction."""
        return self

    def __mul__(self, other: EagerValue) -> "MockEagerTensor":
        """Multiplication."""
        return self

    def __truediv__(self, other: EagerValue) -> "MockEagerTensor":
        """Division."""
        return self

    def __lt__(self, other: EagerValue) -> EagerValue:
        """Less than."""
        return True

    def __le__(self, other: EagerValue) -> EagerValue:
        """Less than or equal."""
        return True

    def __gt__(self, other: EagerValue) -> EagerValue:
        """Greater than."""
        return False

    def __ge__(self, other: EagerValue) -> EagerValue:
        """Greater than or equal."""
        return True

    def __pow__(self, other: EagerValue) -> "MockEagerTensor":
        """Exponentiation."""
        return self


class MockGradTensor:
    """Mock tensor satisfying GradTensorProtocol."""

    def __init__(self) -> None:
        """Initialize MockGradTensor."""
        self._shape = (3, 3)
        self._dtype = "float32"
        self._id = "tensor_grad_0"
        self._data = [1, 2, 3]

    @property
    def dtype(self) -> object:
        """Get dtype."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self._shape

    @property
    def id(self) -> str:
        """Get id."""
        return self._id

    @property
    def data(self) -> object:
        """Get data."""
        return self._data

    def __len__(self) -> int:
        """Get leading dimension length."""
        return 3


class MockInterpreterTensor:
    """Mock tensor satisfying TensorLike interpreter protocol."""

    def __init__(self) -> None:
        """Initialize MockInterpreterTensor."""
        self._shape = (1,)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self._shape

    def __getitem__(self, key: object) -> "MockInterpreterTensor":
        """Get item."""
        return self

    def item(self) -> Union[int, float, bool]:
        """Get scalar item."""
        return 42.0

    def __len__(self) -> int:
        """Get length."""
        return 1


def test_eager_tensor_protocol_conformance() -> None:
    """Verify MockEagerTensor conforms to EagerTensorProtocol."""
    t = MockEagerTensor((4, 4))
    assert isinstance(t, EagerTensorProtocol)
    assert t.shape == (4, 4)
    assert t.dtype == "float32"
    assert len(t) == 4
    assert t[0] is t
    t[0] = t
    assert (-t) is t
    assert (t + 1) is t
    assert (t - 1) is t
    assert (t * 2) is t
    assert (t / 2) is t
    assert (t**2) is t
    assert (t < 5) is True
    assert (t <= 5) is True
    assert (t > 5) is False
    assert (t >= 5) is True


def test_grad_tensor_protocol_conformance() -> None:
    """Verify MockGradTensor conforms to GradTensorProtocol."""
    gt = MockGradTensor()
    assert isinstance(gt, GradTensorProtocol)
    assert gt.shape == (3, 3)
    assert gt.dtype == "float32"
    assert gt.id == "tensor_grad_0"
    assert gt.data == [1, 2, 3]
    assert len(gt) == 3


def test_tensor_like_protocol_conformance() -> None:
    """Verify MockInterpreterTensor conforms to TensorLike protocol."""
    it = MockInterpreterTensor()
    assert isinstance(it, TensorLike)
    assert it.shape == (1,)
    assert it[0] is it
    assert it.item() == 42.0
    assert len(it) == 1


def test_custom_eager_eval_strategy() -> None:
    """Test CustomEagerEvalStrategy evaluate implementation across eager_eval and forward."""

    class OpWithEagerEval:
        """Op class with eager_eval method."""

        def eager_eval(self, a: int, b: int) -> int:
            """Evaluate operation eagerly."""
            return a + b

    class OpWithForward:
        """Op class with forward method."""

        def forward(self, a: int, b: int) -> int:
            """Forward pass."""
            return a * b

    class OpWithoutEval:
        """Op class without eager methods."""

        pass

    class DummyBackend:
        """Dummy backend for testing."""

        @classmethod
        def execute_op(cls, op_type: str, *args: int, **kwargs: int) -> str:
            """Execute op."""
            return f"executed_{op_type}"

    strategy = CustomEagerEvalStrategy()

    ctx_eager = EvaluationContext(
        op_cls=OpWithEagerEval,
        op_type="Add",
        args=[3, 4],
        kwargs={},
        backend=DummyBackend,
    )
    assert strategy.evaluate(ctx_eager) == 7

    ctx_fwd = EvaluationContext(
        op_cls=OpWithForward,
        op_type="Mul",
        args=[3, 4],
        kwargs={},
        backend=DummyBackend,
    )
    assert strategy.evaluate(ctx_fwd) == 12

    ctx_none = EvaluationContext(
        op_cls=OpWithoutEval,
        op_type="NoneOp",
        args=[],
        kwargs={},
        backend=DummyBackend,
    )
    with pytest.raises(NotImplementedError, match="does not implement custom eager evaluation"):
        strategy.evaluate(ctx_none)

    # Test BackendExecuteOpStrategy
    b_strat = BackendExecuteOpStrategy()
    assert b_strat.evaluate(ctx_none) == "executed_NoneOp"

    # Test base abstract evaluate method
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        EvaluationStrategy()  # type: ignore[abstract]

    # Test instance passed as op_cls
    ctx_inst = EvaluationContext(
        op_cls=OpWithEagerEval(),  # instance rather than type
        op_type="Add",
        args=[10, 20],
        kwargs={},
        backend=DummyBackend,
    )
    assert strategy.evaluate(ctx_inst) == 30


def test_eager_evaluator_dispatch_and_packing() -> None:
    """Test EagerEvaluator static evaluate and output packing."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    arr1 = np.array([1.0, 2.0], dtype=np.float32)
    arr2 = np.array([3.0, 4.0], dtype=np.float32)
    t1 = Tensor(arr1, TensorConfig((2,), DType.Float32, "cpu"))
    t2 = Tensor(arr2, TensorConfig((2,), DType.Float32, "cpu"))

    # Test strategy selection
    class CustomOpClass:
        """Custom op class for testing strategy."""

        def eager_eval(self, *args: object, **kwargs: object) -> None:
            """Eager eval."""
            pass

    custom_strat = EagerEvaluator._get_strategy(CustomOpClass)
    assert isinstance(custom_strat, CustomEagerEvalStrategy)

    res = EagerEvaluator.evaluate("Add", t1, t2)
    assert isinstance(res, Tensor)
    np.testing.assert_allclose(res.data, np.array([4.0, 6.0]))

    # Test packing multiple tuple outputs
    res_tuple = EagerEvaluator._pack_outputs((arr1, arr2), t1, "cpu")
    assert isinstance(res_tuple, tuple)
    assert len(res_tuple) == 2
    assert isinstance(res_tuple[0], Tensor)
