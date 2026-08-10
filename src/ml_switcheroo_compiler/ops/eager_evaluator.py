# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Eager evaluation logic."""

import abc
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype


@dataclass
class EvaluationContext:
    """Provide context object for eager evaluation containing operation details."""

    op_cls: Any
    op_type: str
    raw_args: list[Any]
    kwargs: dict[str, Any]
    backend: Any


class EvaluationStrategy(abc.ABC):
    """Define base evaluation strategy."""

    @abc.abstractmethod
    def evaluate(self, ctx: EvaluationContext) -> Any:
        """Evaluate evaluate operation.

        Args:
            ctx (EvaluationContext): The ctx parameter.

        Returns: Any: Result.
        """
        return None


class CustomEagerEvalStrategy(EvaluationStrategy):
    """Strategy for custom eager evaluation."""

    def evaluate(self, ctx: EvaluationContext) -> Any:
        """Evaluate evaluate operation.

        Args:
            ctx (EvaluationContext): Context.

        Returns: Any: Result.
        """
        return ctx.op_cls().eager_eval(*ctx.raw_args, **ctx.kwargs)


class BackendExecuteOpStrategy(EvaluationStrategy):
    """Strategy for backend execution."""

    def evaluate(self, ctx: EvaluationContext) -> Any:
        """Evaluate evaluate operation.

        Args:
            ctx (EvaluationContext): Context.

        Returns: Any: Result.
        """
        return ctx.backend.execute_op(ctx.op_type, *ctx.raw_args, **ctx.kwargs)


class EagerEvaluator:
    """Evaluate operations eagerly."""

    @staticmethod
    def _get_strategy(op_cls: Any) -> EvaluationStrategy:
        """Get the evaluation strategy for a given operation class.

        Args:
            op_cls (object): The operation class.

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
    def _pack_outputs(res_data: Any, first_tensor: Any, device: Any) -> Any:
        """Pack raw output data into Tensor objects.

        Args:
            res_data (object): The raw data returned by the backend.
            first_tensor (object): The first input tensor, used for dtype resolution.
            device (object): The execution device.

        Returns: Any: A Tensor or a tuple of Tensors.
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
        shape = res_data.shape if hasattr(res_data, "shape") else ()
        return Tensor(res_data, TensorConfig(shape, dtype, device))

    @staticmethod
    def evaluate(op_type: str, *args: Any, **kwargs: Any) -> Any:
        """Evaluate evaluate operation.

        Args:
            op_type (str): The op_type parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        from ml_switcheroo_compiler.ops.registry import get_op

        backend = get_active_backend()
        op_cls = get_op(op_type)
        raw_args = [a.data if isinstance(a, Tensor) else a for a in args]

        ctx = EvaluationContext(op_cls, op_type, raw_args, kwargs, backend)
        strategy = EagerEvaluator._get_strategy(op_cls)
        res_data = strategy.evaluate(ctx)

        first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
        device = first_tensor.device if first_tensor is not None else None

        return EagerEvaluator._pack_outputs(res_data, first_tensor, device)
