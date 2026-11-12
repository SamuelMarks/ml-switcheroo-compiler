"""Eager evaluation logic."""

from __future__ import annotations

import abc
import typing

from ml_switcheroo_compiler.backends.registry import ensure_eager_tensor
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    TypedVariadicCallable,
    extract_shape,
    has_backend_member,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype

EagerPayload = TensorOrScalar


class _EagerEvalMethodProtocol(typing.Protocol):
    """Protocol for op classes/instances exposing `eager_eval`."""

    eager_eval: TypedVariadicCallable[EagerPayload]


class _ForwardMethodProtocol(typing.Protocol):
    """Protocol for op instances exposing `forward`."""

    forward: TypedVariadicCallable[EagerPayload]


class _ExecuteOpBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `execute_op`."""

    def execute_op(self, op_type: str, *args: EagerPayload, **kwargs: EagerPayload) -> EagerPayload:
        """Execute operation on backend.

        Args:
            op_type (str): Operation name.
            *args (EagerPayload): Positional operands.
            **kwargs (EagerPayload): Keyword attributes.

        Returns:
            EagerPayload: Result tensor or scalar.
        """
        ...


def _has_eager_eval_method(val: EagerPayload | type) -> typing.TypeGuard[_EagerEvalMethodProtocol]:
    """Check whether `val` exposes `eager_eval`.

    Args:
        val (EagerPayload | type): Operation class or instance.

    Returns:
        bool: True if `val` has `eager_eval`.
    """
    return has_backend_member(val, "eager_eval")


def _has_forward_method(val: EagerPayload | type) -> typing.TypeGuard[_ForwardMethodProtocol]:
    """Check whether `val` exposes `forward`.

    Args:
        val (EagerPayload | type): Operation class or instance.

    Returns:
        bool: True if `val` has `forward`.
    """
    return has_backend_member(val, "forward")


def _has_execute_op_backend(val: EagerPayload | type) -> typing.TypeGuard[_ExecuteOpBackendProtocol]:
    """Check whether `val` exposes `execute_op`.

    Args:
        val (EagerPayload | type): Backend class or instance.

    Returns:
        bool: True if `val` has `execute_op`.
    """
    return has_backend_member(val, "execute_op")


class EvaluationContext:
    """Provide context for eager evaluation containing operation details."""

    def __init__(
        self,
        op_cls: type | EagerPayload | None = None,
        op_type: str = "",
        args: list[EagerPayload] | tuple[EagerPayload, ...] = (),
        kwargs: dict[str, EagerPayload] | None = None,
        backend: type | EagerPayload | None = None,
    ) -> None:
        """Initialize EvaluationContext from positional or keyword arguments.

        Args:
            op_cls (type | EagerPayload | None): Operation class or instance.
            op_type (str): Canonical operation type name.
            args (list[EagerPayload] | tuple[EagerPayload, ...]): Positional tensor or scalar operands.
            kwargs (dict[str, EagerPayload] | None): Keyword operation attributes.
            backend (type | EagerPayload | None): Target eager backend class or module.
        """
        self.op_cls: type | EagerPayload | None = op_cls
        self.op_type: str = op_type
        self.args: list[EagerPayload] | tuple[EagerPayload, ...] = args
        self.kwargs: dict[str, EagerPayload] = {} if kwargs is None else kwargs
        self.backend: type | EagerPayload | None = backend


class EvaluationStrategy(abc.ABC):
    """Define base evaluation strategy."""

    @abc.abstractmethod
    def evaluate(self, ctx: EvaluationContext) -> EagerPayload:
        """Evaluate operation within evaluation context.

        Args:
            ctx (EvaluationContext): The evaluation context.

        Returns:
            EagerPayload: Evaluated raw tensor or scalar data.
        """
        raise NotImplementedError


class CustomEagerEvalStrategy(EvaluationStrategy):
    """Strategy for custom eager evaluation."""

    def evaluate(self, ctx: EvaluationContext) -> EagerPayload:
        """Evaluate operation using custom eager execution logic.

        Args:
            ctx (EvaluationContext): Context.

        Returns:
            EagerPayload: Evaluated raw tensor or scalar data.

        Raises:
            NotImplementedError: If neither eager_eval nor forward is callable.
        """
        op_cls = ctx.op_cls
        op_inst: EagerPayload = op_cls() if isinstance(op_cls, type) else op_cls
        if _has_eager_eval_method(op_inst) and callable(op_inst.eager_eval):
            return ensure_eager_tensor(op_inst.eager_eval(*ctx.args, **ctx.kwargs))
        if _has_forward_method(op_inst) and callable(op_inst.forward):
            return ensure_eager_tensor(op_inst.forward(*ctx.args, **ctx.kwargs))
        raise NotImplementedError(f"Operation class '{ctx.op_type}' does not implement custom eager evaluation.")


class BackendExecuteOpStrategy(EvaluationStrategy):
    """Strategy for backend execution."""

    def evaluate(self, ctx: EvaluationContext) -> EagerPayload:
        """Evaluate operation via backend execute_op dispatch.

        Args:
            ctx (EvaluationContext): Context.

        Returns:
            EagerPayload: Evaluated raw tensor or scalar data.
        """
        raw_backend: EagerPayload = ctx.backend
        assert _has_execute_op_backend(raw_backend)
        return ensure_eager_tensor(raw_backend.execute_op(ctx.op_type, *ctx.args, **ctx.kwargs))


class EagerEvaluator:
    """Evaluate operations eagerly."""

    @staticmethod
    def _get_strategy(op_cls: type) -> EvaluationStrategy:
        """Get the evaluation strategy for a given operation class.

        Args:
            op_cls (type): The operation class.

        Returns:
            EvaluationStrategy: The determined evaluation strategy.
        """
        from ml_switcheroo_compiler.ops.base import OpDef

        raw_op_cls: type = op_cls
        raw_base: type = OpDef
        base_eval = raw_base.eager_eval if _has_eager_eval_method(raw_base) else None
        has_custom_eval = _has_eager_eval_method(raw_op_cls) and raw_op_cls.eager_eval is not base_eval
        if has_custom_eval:
            return CustomEagerEvalStrategy()
        return BackendExecuteOpStrategy()

    @staticmethod
    def _pack_outputs(
        res_data: EagerPayload,
        first_tensor: Tensor | None,
        device: Device | str | None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Pack raw output data into Tensor objects.

        Args:
            res_data (EagerPayload): The raw data returned by the backend.
            first_tensor (Tensor | None): The first input tensor, used for dtype resolution.
            device (Device | str | None): The execution device.

        Returns: Tensor: A Tensor or a tuple of Tensors.
        """
        if isinstance(res_data, (tuple, list)):
            return tuple(
                Tensor(
                    d,
                    TensorConfig(
                        extract_shape(d),
                        resolve_dtype(d, first_tensor),
                        device,
                    ),
                )
                for d in res_data
            )

        dtype = resolve_dtype(res_data, first_tensor)
        shape = extract_shape(res_data)
        return Tensor(res_data, TensorConfig(shape, dtype, device))

    @staticmethod
    def evaluate(op_type: str, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | tuple[Tensor, ...]:
        """Execute the eager evaluate operation on the active backend array.

        Args:
            op_type (str): Input `op_type` argument (op type).
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        from ml_switcheroo_compiler.ops.registry import get_op

        backend = get_active_backend()
        op_cls = get_op(op_type)
        raw_args: list[EagerPayload] = [a.data if isinstance(a, Tensor) else a for a in args]

        ctx = EvaluationContext(op_cls=op_cls, op_type=op_type, args=raw_args, kwargs=dict(kwargs), backend=backend)
        strategy = EagerEvaluator._get_strategy(op_cls)
        res_data = strategy.evaluate(ctx)

        first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
        device = first_tensor.device if first_tensor is not None else None

        return EagerEvaluator._pack_outputs(res_data, first_tensor, device)
