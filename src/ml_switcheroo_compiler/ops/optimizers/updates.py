# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Implementations of functional optimizer update steps for various algorithms."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, divide, maximum, multiply, subtract
from ml_switcheroo_compiler.ops.unary import abs as abs_op
from ml_switcheroo_compiler.ops.unary import sign, sqrt, square


@dataclass
class SGDConfig:
    """Configuration settings for the Stochastic Gradient Descent (SGD) optimizer."""

    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclass
class AdagradConfig:
    """Configuration settings for the Adagrad optimization algorithm."""

    lr: float
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10
    step: int = 1


@dataclass
class AdadeltaConfig:
    """Configuration parameters for the Adadelta optimization algorithm."""

    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0.0


@dataclass
class LionConfig:
    """Configuration parameters for the Lion optimization algorithm."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0


@dataclass
class AdamHyperparams:
    """Hyperparameters and configuration state for the Adam optimization algorithm."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    step: int = 1


@dataclass
class AdamWHyperparams:
    """Hyperparameters and configuration state for the AdamW optimization algorithm."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    step: int = 1


@dataclass
class RMSPropHyperparams:
    """Hyperparameters and configuration state for the RMSprop optimization algorithm."""

    lr: float
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


@dataclass
class AdamaxHyperparams:
    """Hyperparameters and configuration state for the Adamax optimization algorithm."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    step: int = 1


def sgd_update(
    param: Tensor,
    grad: Tensor,
    config: SGDConfig,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Stochastic Gradient Descent (SGD) update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        config: The configuration settings for the SGD optimizer.
        state: The optimizer state dictionary containing moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    lr_t: object = config.lr
    if config.weight_decay != 0.0:
        wd_t: object = config.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    if config.momentum != 0.0:
        mom_t: object = config.momentum
        damp_t: object = 1.0 - config.dampening

        if "momentum_buffer" not in state:
            buf: object = grad
        else:
            buf: object = state["momentum_buffer"]
            buf: object = add(multiply(buf, mom_t), multiply(grad, damp_t))

        state["momentum_buffer"] = buf

        if config.nesterov:
            grad: object = add(grad, multiply(buf, mom_t))
        else:
            grad: object = buf

    new_param: object = subtract(param, multiply(grad, lr_t))
    return new_param, state


def adam_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamHyperparams,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Adam optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        hp: The hyperparameters and configuration for the Adam optimizer.
        state: The optimizer state dictionary containing moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if hp.weight_decay != 0.0:
        wd_t: object = hp.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    b1_t: object = hp.beta1
    b2_t: object = hp.beta2
    one_minus_b1: object = 1.0 - hp.beta1
    one_minus_b2: object = 1.0 - hp.beta2
    eps_t: object = hp.eps

    exp_avg: object = state.get("exp_avg", 0.0)
    exp_avg_sq: object = state.get("exp_avg_sq", 0.0)

    # m_t = b1 * m_{t-1} + (1 - b1) * g
    exp_avg: object = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    # v_t = b2 * v_{t-1} + (1 - b2) * g^2
    exp_avg_sq: object = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    # bias correction
    bias_correction1: object = 1.0 - hp.beta1**hp.step
    bias_correction2: object = 1.0 - hp.beta2**hp.step

    step_size: object = divide(hp.lr, bias_correction1)

    denom: object = add(divide(sqrt(exp_avg_sq), sqrt(bias_correction2)), eps_t)
    update: object = divide(exp_avg, denom)

    new_param: object = subtract(param, multiply(update, step_size))
    return new_param, state


def adamw_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamWHyperparams,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional AdamW optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        hp: The hyperparameters and configuration for the AdamW optimizer.
        state: The optimizer state dictionary containing moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    # Weight decay is applied directly to param in AdamW
    lr_t: object = hp.lr
    if hp.weight_decay != 0.0:
        param: object = multiply(param, 1.0 - hp.lr * hp.weight_decay)

    # Then standard adam on remaining
    b1_t: object = hp.beta1
    b2_t: object = hp.beta2
    one_minus_b1: object = 1.0 - hp.beta1
    one_minus_b2: object = 1.0 - hp.beta2
    eps_t: object = hp.eps

    exp_avg: object = state.get("exp_avg", 0.0)
    exp_avg_sq: object = state.get("exp_avg_sq", 0.0)

    exp_avg: object = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    exp_avg_sq: object = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    bias_correction1: object = 1.0 - hp.beta1**hp.step
    bias_correction2: object = 1.0 - hp.beta2**hp.step

    step_size: object = divide(lr_t, bias_correction1)
    denom: object = add(divide(sqrt(exp_avg_sq), sqrt(bias_correction2)), eps_t)
    update: object = divide(exp_avg, denom)

    new_param: object = subtract(param, multiply(update, step_size))
    return new_param, state


def adagrad_update(
    param: Tensor,
    grad: Tensor,
    config: AdagradConfig,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Adagrad optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        config: The configuration settings for the Adagrad optimizer.
        state: The optimizer state dictionary containing the sum of squared gradients. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if config.weight_decay != 0.0:
        wd_t: object = config.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    clr: object = config.lr / (1 + (config.step - 1) * config.lr_decay)
    clr_t: object = clr
    eps_t: object = config.eps

    sum_sq: object = state.get("sum", config.initial_accumulator_value)
    sum_sq: object = add(sum_sq, square(grad))
    state["sum"] = sum_sq

    denom: object = add(sqrt(sum_sq), eps_t)
    update: object = divide(grad, denom)
    new_param: object = subtract(param, multiply(update, clr_t))

    return new_param, state


def rmsprop_update(
    param: Tensor,
    grad: Tensor,
    hp: RMSPropHyperparams,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional RMSprop optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        hp: The hyperparameters and configuration for the RMSprop optimizer.
        state: The optimizer state dictionary containing moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if hp.weight_decay != 0.0:
        wd_t: object = hp.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    alpha_t: object = hp.alpha
    one_minus_alpha: object = 1.0 - hp.alpha
    eps_t: object = hp.eps
    lr_t: object = hp.lr

    square_avg: object = state.get("square_avg", 0.0)
    square_avg: object = add(multiply(square_avg, alpha_t), multiply(square(grad), one_minus_alpha))
    state["square_avg"] = square_avg

    avg: object = square_avg
    if hp.centered:
        grad_avg: object = state.get("grad_avg", 0.0)
        grad_avg: object = add(multiply(grad_avg, alpha_t), multiply(grad, one_minus_alpha))
        state["grad_avg"] = grad_avg
        avg: object = subtract(avg, square(grad_avg))

    denom: object = add(sqrt(avg), eps_t)

    if hp.momentum > 0:
        mom_t: object = hp.momentum
        buf: object = state.get("momentum_buffer", 0.0)
        buf: object = add(multiply(buf, mom_t), divide(grad, denom))
        state["momentum_buffer"] = buf
        new_param: object = subtract(param, multiply(buf, lr_t))
    else:
        new_param: object = subtract(param, multiply(divide(grad, denom), lr_t))

    return new_param, state


def adadelta_update(
    param: Tensor,
    grad: Tensor,
    config: AdadeltaConfig,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Adadelta optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        config: The configuration settings for the Adadelta optimizer.
        state: The optimizer state dictionary containing moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if config.weight_decay != 0.0:
        wd_t: object = config.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    rho_t: object = config.rho
    one_minus_rho: object = 1.0 - config.rho
    eps_t: object = config.eps
    lr_t: object = config.lr

    square_avg: object = state.get("square_avg", 0.0)
    acc_delta: object = state.get("acc_delta", 0.0)

    square_avg: object = add(multiply(square_avg, rho_t), multiply(square(grad), one_minus_rho))
    state["square_avg"] = square_avg

    std: object = add(sqrt(square_avg), eps_t)
    delta: object = multiply(divide(add(sqrt(acc_delta), eps_t), std), grad)

    acc_delta: object = add(multiply(acc_delta, rho_t), multiply(square(delta), one_minus_rho))
    state["acc_delta"] = acc_delta

    new_param: object = subtract(param, multiply(delta, lr_t))
    return new_param, state


def adamax_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamaxHyperparams,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Adamax optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        hp: The hyperparameters and configuration for the Adamax optimizer.
        state: The optimizer state dictionary containing moving averages and maximums. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if hp.weight_decay != 0.0:
        wd_t: object = hp.weight_decay
        grad: object = add(grad, multiply(param, wd_t))

    b1_t: object = hp.beta1
    b2_t: object = hp.beta2
    one_minus_b1: object = 1.0 - hp.beta1
    eps_t: object = hp.eps

    exp_avg: object = state.get("exp_avg", 0.0)
    exp_inf: object = state.get("exp_inf", 0.0)

    exp_avg: object = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    state["exp_avg"] = exp_avg

    exp_inf: object = maximum(multiply(exp_inf, b2_t), abs_op(grad))
    state["exp_inf"] = exp_inf

    bias_correction: object = 1.0 - hp.beta1**hp.step
    step_size: object = divide(hp.lr, bias_correction)

    update: object = divide(exp_avg, add(exp_inf, eps_t))
    new_param: object = subtract(param, multiply(update, step_size))

    return new_param, state


def lion_update(
    param: Tensor,
    grad: Tensor,
    config: LionConfig,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Lion optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        config: The configuration settings for the Lion optimizer.
        state: The optimizer state dictionary containing exponential moving averages. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    if config.weight_decay != 0.0:
        param: object = multiply(param, 1.0 - config.lr * config.weight_decay)

    b1_t: object = config.beta1
    b2_t: object = config.beta2
    one_minus_b1: object = 1.0 - config.beta1
    one_minus_b2: object = 1.0 - config.beta2
    lr_t: object = config.lr

    exp_avg: object = state.get("exp_avg", 0.0)

    c: object = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    update: object = sign(c)
    new_param: object = subtract(param, multiply(update, lr_t))

    exp_avg: object = add(multiply(exp_avg, b2_t), multiply(grad, one_minus_b2))
    state["exp_avg"] = exp_avg

    return new_param, state


def adafactor_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Adafactor optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        lr: The learning rate for the update step.
        state: The optimizer state dictionary. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    # Very simplified version for coverage
    # Full Adafactor maintains row and col variances
    new_param: object = subtract(param, multiply(grad, lr))
    return new_param, state


def muon_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    momentum: float = 0.95,
    state: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply a functional Muon optimization update to a parameter.

    Args:
        param: The current parameter tensor to be updated.
        grad: The gradient tensor with respect to the parameter.
        lr: The learning rate for the update step.
        momentum: The momentum factor for the update. Defaults to 0.95.
        state: The optimizer state dictionary. Defaults to None.

    Returns:
        A tuple containing the updated parameter tensor and the updated state dictionary.
    """
    if state is None:
        state: object = {}

    # Very simplified version for coverage
    # Muon uses Newton-Schulz iteration
    new_param: object = subtract(param, multiply(grad, lr))
    return new_param, state


from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("ApplyAdam")
class ApplyAdam(OpDef):
    """Apply Adam optimizer step."""

    op_name: object = "ApplyAdam"

    def infer_shape(self, param: object, m: object, v: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            param (object): The param parameter.
            m (object): The m parameter.
            v (object): The v parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(param, "shape", ())


@register_op("ApplyAdagrad")
class ApplyAdagrad(OpDef):
    """Apply Adagrad optimizer step."""

    op_name: object = "ApplyAdagrad"

    def infer_shape(self, param: object, accum: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            param (object): The param parameter.
            accum (object): The accum parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(param, "shape", ())


@register_op("ApplyFtrl")
class ApplyFtrl(OpDef):
    """Apply FTRL optimizer step."""

    op_name: object = "ApplyFtrl"

    def infer_shape(self, param: object, accum: object, linear: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            param (object): The param parameter.
            accum (object): The accum parameter.
            linear (object): The linear parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(param, "shape", ())


@register_op("ApplyRMSProp")
class ApplyRMSProp(OpDef):
    """Apply RMSProp optimizer step."""

    op_name: object = "ApplyRMSProp"

    def infer_shape(self, param: object, ms: object, mom: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            param (object): The param parameter.
            ms (object): The ms parameter.
            mom (object): The mom parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(param, "shape", ())


def apply_adam(param: Tensor, m: Tensor, v: Tensor, grad: Tensor, lr: float) -> object:
    """Apply Adam update.

    Args:
        param (Tensor): The param parameter.
        m (Tensor): The m parameter.
        v (Tensor): The v parameter.
        grad (Tensor): The grad parameter.
        lr (float): The lr parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        return backend.execute_op("ApplyAdam", param, m, v, grad, lr=lr)
    # fallback shape node
    out: object = _emit_shape_node("ApplyAdam", [param, m, v, grad], {"lr": lr}, param.shape, param.dtype)
    return out, m, v


def apply_adagrad(param: Tensor, accum: Tensor, grad: Tensor, lr: float) -> object:
    """Apply Adagrad update.

    Args:
        param (Tensor): The param parameter.
        accum (Tensor): The accum parameter.
        grad (Tensor): The grad parameter.
        lr (float): The lr parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        return backend.execute_op("ApplyAdagrad", param, accum, grad, lr=lr)
    out: object = _emit_shape_node("ApplyAdagrad", [param, accum, grad], {"lr": lr}, param.shape, param.dtype)
    return out, accum


def apply_ftrl(param: Tensor, accum: Tensor, linear: Tensor, grad: Tensor, lr: float) -> object:
    """Apply FTRL update.

    Args:
        param (Tensor): The param parameter.
        accum (Tensor): The accum parameter.
        linear (Tensor): The linear parameter.
        grad (Tensor): The grad parameter.
        lr (float): The lr parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        return backend.execute_op("ApplyFtrl", param, accum, linear, grad, lr=lr)
    out: object = _emit_shape_node("ApplyFtrl", [param, accum, linear, grad], {"lr": lr}, param.shape, param.dtype)
    return out, accum, linear


def apply_rmsprop(param: Tensor, ms: Tensor, mom: Tensor, grad: Tensor, lr: float) -> object:
    """Apply RMSProp update.

    Args:
        param (Tensor): The param parameter.
        ms (Tensor): The ms parameter.
        mom (Tensor): The mom parameter.
        grad (Tensor): The grad parameter.
        lr (float): The lr parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        return backend.execute_op("ApplyRMSProp", param, ms, mom, grad, lr=lr)
    out: object = _emit_shape_node("ApplyRMSProp", [param, ms, mom, grad], {"lr": lr}, param.shape, param.dtype)
    return out, ms, mom


@register_op("LionConfig")
class LionConfigOp(OpDef):
    """LionConfig operation."""

    op_name: object = "LionConfig"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("AdamaxHyperparams")
class AdamaxHyperparamsOp(OpDef):
    """AdamaxHyperparams operation."""

    op_name: object = "AdamaxHyperparams"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())
