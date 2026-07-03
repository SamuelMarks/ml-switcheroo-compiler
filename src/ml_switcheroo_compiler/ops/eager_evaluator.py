"""Eager evaluation logic."""

from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype


@dataclass
class EvaluationContext:
    """Context object for eager evaluation containing operation details."""

    op_cls: object
    op_type: str
    raw_args: list[Any]
    kwargs: dict[str, Any]
    backend: object


class EvaluationStrategy:
    """Base evaluation strategy."""

    def evaluate(self, ctx: EvaluationContext) -> object:
        """Evaluate."""
        raise NotImplementedError  # pragma: no cover


class CustomEagerEvalStrategy(EvaluationStrategy):
    """Strategy for custom eager evaluation."""

    def evaluate(self, ctx: EvaluationContext) -> object:
        """Evaluate."""
        return ctx.op_cls().eager_eval(*ctx.raw_args, **ctx.kwargs)


class BackendExecuteOpStrategy(EvaluationStrategy):
    """Strategy for backend execution."""

    def evaluate(self, ctx: EvaluationContext) -> object:
        """Evaluate."""
        return ctx.backend.execute_op(ctx.op_type, *ctx.raw_args, **ctx.kwargs)  # pragma: no cover


class EagerEvaluator:
    """Evaluates operations eagerly."""

    @staticmethod
    def _get_strategy(op_cls: object) -> EvaluationStrategy:
        """Get the evaluation strategy."""
        has_custom_eval = hasattr(op_cls, "eager_eval") and op_cls.__dict__.get("eager_eval") is not getattr(  # pragma: no branch
            __import__("ml_switcheroo_compiler.ops.base", fromlist=["OpDef"]).OpDef,
            "eager_eval",
            None,
        )
        if has_custom_eval:
            return CustomEagerEvalStrategy()
        return BackendExecuteOpStrategy()  # pragma: no cover

    @staticmethod
    def evaluate(op_type: str, *args: object, **kwargs: object) -> object:
        """Docstring."""
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        from ml_switcheroo_compiler.ops.registry import get_op

        op_cls = get_op(op_type)
        raw_args = [a.data if isinstance(a, Tensor) else a for a in args]

        ctx = EvaluationContext(op_cls, op_type, raw_args, kwargs, backend)
        strategy = EagerEvaluator._get_strategy(op_cls)
        res_data = strategy.evaluate(ctx)

        first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
        device = first_tensor.device if first_tensor is not None else None

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
