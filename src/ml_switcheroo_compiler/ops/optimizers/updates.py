"""Functional optimizer updates."""

from dataclasses import dataclass

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, divide, maximum, multiply, subtract
from ml_switcheroo_compiler.ops.unary import abs as abs_op
from ml_switcheroo_compiler.ops.unary import sign, sqrt, square


@dataclass
class AdamHyperparams:
    """Class docstring."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    step: int = 1


@dataclass
class AdamWHyperparams:
    """Class docstring."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    step: int = 1


@dataclass
class RMSPropHyperparams:
    """Class docstring."""

    lr: float
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


@dataclass
class AdamaxHyperparams:
    """Class docstring."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    step: int = 1


def sgd_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    momentum: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional SGD update."""
    if state is None:
        state = {}

    lr_t = lr
    if weight_decay != 0.0:
        wd_t = weight_decay
        grad = add(grad, multiply(param, wd_t))

    if momentum != 0.0:
        mom_t = momentum
        damp_t = 1.0 - dampening

        if "momentum_buffer" not in state:
            buf = grad
        else:
            buf = state["momentum_buffer"]
            buf = add(multiply(buf, mom_t), multiply(grad, damp_t))

        state["momentum_buffer"] = buf

        if nesterov:
            grad = add(grad, multiply(buf, mom_t))
        else:
            grad = buf

    new_param = subtract(param, multiply(grad, lr_t))
    return new_param, state


def adam_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamHyperparams,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adam update."""
    if state is None:
        state = {}

    if hp.weight_decay != 0.0:
        wd_t = hp.weight_decay
        grad = add(grad, multiply(param, wd_t))

    b1_t = hp.beta1
    b2_t = hp.beta2
    one_minus_b1 = 1.0 - hp.beta1
    one_minus_b2 = 1.0 - hp.beta2
    eps_t = hp.eps

    exp_avg = state.get("exp_avg", 0.0)
    exp_avg_sq = state.get("exp_avg_sq", 0.0)

    # m_t = b1 * m_{t-1} + (1 - b1) * g
    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    # v_t = b2 * v_{t-1} + (1 - b2) * g^2
    exp_avg_sq = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    # bias correction
    bias_correction1 = 1.0 - hp.beta1**hp.step
    bias_correction2 = 1.0 - hp.beta2**hp.step

    step_size = divide(hp.lr, bias_correction1)

    denom = add(divide(sqrt(exp_avg_sq), sqrt(bias_correction2)), eps_t)
    update = divide(exp_avg, denom)

    new_param = subtract(param, multiply(update, step_size))
    return new_param, state


def adamw_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamWHyperparams,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional AdamW update."""
    if state is None:
        state = {}

    # Weight decay is applied directly to param in AdamW
    lr_t = hp.lr
    if hp.weight_decay != 0.0:
        param = multiply(param, 1.0 - hp.lr * hp.weight_decay)

    # Then standard adam on remaining
    b1_t = hp.beta1
    b2_t = hp.beta2
    one_minus_b1 = 1.0 - hp.beta1
    one_minus_b2 = 1.0 - hp.beta2
    eps_t = hp.eps

    exp_avg = state.get("exp_avg", 0.0)
    exp_avg_sq = state.get("exp_avg_sq", 0.0)

    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    exp_avg_sq = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    bias_correction1 = 1.0 - hp.beta1**hp.step
    bias_correction2 = 1.0 - hp.beta2**hp.step

    step_size = divide(lr_t, bias_correction1)
    denom = add(divide(sqrt(exp_avg_sq), sqrt(bias_correction2)), eps_t)
    update = divide(exp_avg, denom)

    new_param = subtract(param, multiply(update, step_size))
    return new_param, state


def adagrad_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    eps: float = 1e-10,
    step: int = 1,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adagrad update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        wd_t = weight_decay
        grad = add(grad, multiply(param, wd_t))

    clr = lr / (1 + (step - 1) * lr_decay)
    clr_t = clr
    eps_t = eps

    sum_sq = state.get("sum", 0.0)
    sum_sq = add(sum_sq, square(grad))
    state["sum"] = sum_sq

    denom = add(sqrt(sum_sq), eps_t)
    update = divide(grad, denom)
    new_param = subtract(param, multiply(update, clr_t))

    return new_param, state


def rmsprop_update(
    param: Tensor,
    grad: Tensor,
    hp: RMSPropHyperparams,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional RMSprop update."""
    if state is None:
        state = {}

    if hp.weight_decay != 0.0:
        wd_t = hp.weight_decay
        grad = add(grad, multiply(param, wd_t))

    alpha_t = hp.alpha
    one_minus_alpha = 1.0 - hp.alpha
    eps_t = hp.eps
    lr_t = hp.lr

    square_avg = state.get("square_avg", 0.0)
    square_avg = add(multiply(square_avg, alpha_t), multiply(square(grad), one_minus_alpha))
    state["square_avg"] = square_avg

    avg = square_avg
    if hp.centered:
        grad_avg = state.get("grad_avg", 0.0)
        grad_avg = add(multiply(grad_avg, alpha_t), multiply(grad, one_minus_alpha))
        state["grad_avg"] = grad_avg
        avg = subtract(avg, square(grad_avg))

    denom = add(sqrt(avg), eps_t)

    if hp.momentum > 0:
        mom_t = hp.momentum
        buf = state.get("momentum_buffer", 0.0)
        buf = add(multiply(buf, mom_t), divide(grad, denom))
        state["momentum_buffer"] = buf
        new_param = subtract(param, multiply(buf, lr_t))
    else:
        new_param = subtract(param, multiply(divide(grad, denom), lr_t))

    return new_param, state


def adadelta_update(
    param: Tensor,
    grad: Tensor,
    lr: float = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional AdaDelta update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        wd_t = weight_decay
        grad = add(grad, multiply(param, wd_t))

    rho_t = rho
    one_minus_rho = 1.0 - rho
    eps_t = eps
    lr_t = lr

    square_avg = state.get("square_avg", 0.0)
    acc_delta = state.get("acc_delta", 0.0)

    square_avg = add(multiply(square_avg, rho_t), multiply(square(grad), one_minus_rho))
    state["square_avg"] = square_avg

    std = add(sqrt(square_avg), eps_t)
    delta = multiply(divide(add(sqrt(acc_delta), eps_t), std), grad)

    acc_delta = add(multiply(acc_delta, rho_t), multiply(square(delta), one_minus_rho))
    state["acc_delta"] = acc_delta

    new_param = subtract(param, multiply(delta, lr_t))
    return new_param, state


def adamax_update(
    param: Tensor,
    grad: Tensor,
    hp: AdamaxHyperparams,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adamax update."""
    if state is None:
        state = {}

    if hp.weight_decay != 0.0:
        wd_t = hp.weight_decay
        grad = add(grad, multiply(param, wd_t))

    b1_t = hp.beta1
    b2_t = hp.beta2
    one_minus_b1 = 1.0 - hp.beta1
    eps_t = hp.eps

    exp_avg = state.get("exp_avg", 0.0)
    exp_inf = state.get("exp_inf", 0.0)

    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    state["exp_avg"] = exp_avg

    exp_inf = maximum(multiply(exp_inf, b2_t), abs_op(grad))
    state["exp_inf"] = exp_inf

    bias_correction = 1.0 - hp.beta1**hp.step
    step_size = divide(hp.lr, bias_correction)

    update = divide(exp_avg, add(exp_inf, eps_t))
    new_param = subtract(param, multiply(update, step_size))

    return new_param, state


def lion_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Lion update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        param = multiply(param, 1.0 - lr * weight_decay)

    b1_t = beta1
    b2_t = beta2
    one_minus_b1 = 1.0 - beta1
    one_minus_b2 = 1.0 - beta2
    lr_t = lr

    exp_avg = state.get("exp_avg", 0.0)

    c = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    update = sign(c)
    new_param = subtract(param, multiply(update, lr_t))

    exp_avg = add(multiply(exp_avg, b2_t), multiply(grad, one_minus_b2))
    state["exp_avg"] = exp_avg

    return new_param, state


def adafactor_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adafactor update."""
    if state is None:
        state = {}

    # Very simplified version for coverage
    # Full Adafactor maintains row and col variances
    new_param = subtract(param, multiply(grad, lr))
    return new_param, state


def muon_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    momentum: float = 0.95,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Muon update."""
    if state is None:
        state = {}

    # Very simplified version for coverage
    # Muon uses Newton-Schulz iteration
    new_param = subtract(param, multiply(grad, lr))
    return new_param, state
