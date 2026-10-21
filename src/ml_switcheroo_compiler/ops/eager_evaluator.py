"""Eager evaluation logic."""

from __future__ import annotations

import abc
import typing  # noqa: F401

import pydantic

from ml_switcheroo_compiler.backends.registry import as_tensor
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype

EagerPayload = TensorOrScalar


class EvaluationContext(pydantic.BaseModel):
    """Provide context for eager evaluation containing operation details."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    op_cls: type | EagerPayload | None = None
    op_type: str = ""
    args: list[EagerPayload] | tuple[EagerPayload, ...] = ()
    kwargs: dict[str, EagerPayload] = pydantic.Field(default_factory=dict)
    backend: type | EagerPayload | None = None

    def __init__(self, *args: EagerPayload, **kwargs: EagerPayload) -> None:
        """Initialize EvaluationContext from positional or keyword arguments."""
        super().__init__(**{**dict(zip(self.__class__.model_fields.keys(), args)), **kwargs})


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
        op_inst = op_cls() if isinstance(op_cls, type) else op_cls
        eager_eval_fn = getattr(op_inst, "eager_eval", None)
        if callable(eager_eval_fn):
            return as_tensor(eager_eval_fn(*ctx.args, **ctx.kwargs))
        forward_fn = getattr(op_inst, "forward", None)
        if callable(forward_fn):
            return as_tensor(forward_fn(*ctx.args, **ctx.kwargs))
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
        return as_tensor(getattr(ctx.backend, "execute_op")(ctx.op_type, *ctx.args, **ctx.kwargs))


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

        has_custom_eval = hasattr(op_cls, "eager_eval") and op_cls.__dict__.get("eager_eval") is not getattr(
            OpDef,
            "eager_eval",
            None,
        )
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
                        d.shape if hasattr(d, "shape") else (),
                        resolve_dtype(d, first_tensor),
                        device,
                    ),
                )
                for d in res_data
            )

        dtype = resolve_dtype(res_data, first_tensor)
        shape = as_tensor(res_data).shape if hasattr(res_data, "shape") else ()
        return Tensor(res_data, TensorConfig(shape, dtype, device))

    @staticmethod
    def evaluate(op_type: str, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor | tuple[Tensor, ...]:
        """Execute the eager evaluate operation on the active backend array.

        Args:
            op_type (str): The op_type parameter.
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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
