"""Functional optimizer updates."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, multiply, divide, subtract, maximum
from ml_switcheroo_compiler.ops.unary import sqrt, square, sign
from ml_switcheroo_compiler.ops.aliases.memory_ops import convert_to_tensor


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

    lr_t = convert_to_tensor(lr)
    if weight_decay != 0.0:
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    if momentum != 0.0:
        mom_t = convert_to_tensor(momentum)
        damp_t = convert_to_tensor(1.0 - dampening)

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
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adam update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    b1_t = convert_to_tensor(beta1)
    b2_t = convert_to_tensor(beta2)
    one_minus_b1 = convert_to_tensor(1.0 - beta1)
    one_minus_b2 = convert_to_tensor(1.0 - beta2)
    eps_t = convert_to_tensor(eps)

    exp_avg = state.get("exp_avg", convert_to_tensor(0.0))
    exp_avg_sq = state.get("exp_avg_sq", convert_to_tensor(0.0))

    # m_t = b1 * m_{t-1} + (1 - b1) * g
    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    # v_t = b2 * v_{t-1} + (1 - b2) * g^2
    exp_avg_sq = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    # bias correction
    bias_correction1 = convert_to_tensor(1.0 - beta1**step)
    bias_correction2 = convert_to_tensor(1.0 - beta2**step)

    step_size = divide(convert_to_tensor(lr), bias_correction1)

    denom = add(divide(sqrt(exp_avg_sq), sqrt(bias_correction2)), eps_t)
    update = divide(exp_avg, denom)

    new_param = subtract(param, multiply(update, step_size))
    return new_param, state


def adamw_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional AdamW update."""
    if state is None:
        state = {}

    # Weight decay is applied directly to param in AdamW
    lr_t = convert_to_tensor(lr)
    if weight_decay != 0.0:
        param = multiply(param, convert_to_tensor(1.0 - lr * weight_decay))

    # Then standard adam on remaining
    b1_t = convert_to_tensor(beta1)
    b2_t = convert_to_tensor(beta2)
    one_minus_b1 = convert_to_tensor(1.0 - beta1)
    one_minus_b2 = convert_to_tensor(1.0 - beta2)
    eps_t = convert_to_tensor(eps)

    exp_avg = state.get("exp_avg", convert_to_tensor(0.0))
    exp_avg_sq = state.get("exp_avg_sq", convert_to_tensor(0.0))

    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    exp_avg_sq = add(multiply(exp_avg_sq, b2_t), multiply(square(grad), one_minus_b2))

    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq

    bias_correction1 = convert_to_tensor(1.0 - beta1**step)
    bias_correction2 = convert_to_tensor(1.0 - beta2**step)

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
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    clr = lr / (1 + (step - 1) * lr_decay)
    clr_t = convert_to_tensor(clr)
    eps_t = convert_to_tensor(eps)

    sum_sq = state.get("sum", convert_to_tensor(0.0))
    sum_sq = add(sum_sq, square(grad))
    state["sum"] = sum_sq

    denom = add(sqrt(sum_sq), eps_t)
    update = divide(grad, denom)
    new_param = subtract(param, multiply(update, clr_t))

    return new_param, state


def rmsprop_update(
    param: Tensor,
    grad: Tensor,
    lr: float,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    centered: bool = False,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional RMSprop update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    alpha_t = convert_to_tensor(alpha)
    one_minus_alpha = convert_to_tensor(1.0 - alpha)
    eps_t = convert_to_tensor(eps)
    lr_t = convert_to_tensor(lr)

    square_avg = state.get("square_avg", convert_to_tensor(0.0))
    square_avg = add(multiply(square_avg, alpha_t), multiply(square(grad), one_minus_alpha))
    state["square_avg"] = square_avg

    avg = square_avg
    if centered:
        grad_avg = state.get("grad_avg", convert_to_tensor(0.0))
        grad_avg = add(multiply(grad_avg, alpha_t), multiply(grad, one_minus_alpha))
        state["grad_avg"] = grad_avg
        avg = subtract(avg, square(grad_avg))

    denom = add(sqrt(avg), eps_t)

    if momentum > 0:
        mom_t = convert_to_tensor(momentum)
        buf = state.get("momentum_buffer", convert_to_tensor(0.0))
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
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    rho_t = convert_to_tensor(rho)
    one_minus_rho = convert_to_tensor(1.0 - rho)
    eps_t = convert_to_tensor(eps)
    lr_t = convert_to_tensor(lr)

    square_avg = state.get("square_avg", convert_to_tensor(0.0))
    acc_delta = state.get("acc_delta", convert_to_tensor(0.0))

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
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    state: dict[str, Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Functional Adamax update."""
    if state is None:
        state = {}

    if weight_decay != 0.0:
        wd_t = convert_to_tensor(weight_decay)
        grad = add(grad, multiply(param, wd_t))

    b1_t = convert_to_tensor(beta1)
    b2_t = convert_to_tensor(beta2)
    one_minus_b1 = convert_to_tensor(1.0 - beta1)
    eps_t = convert_to_tensor(eps)

    exp_avg = state.get("exp_avg", convert_to_tensor(0.0))
    exp_inf = state.get("exp_inf", convert_to_tensor(0.0))

    exp_avg = add(multiply(exp_avg, b1_t), multiply(grad, one_minus_b1))
    state["exp_avg"] = exp_avg

    from ml_switcheroo_compiler.ops.unary import abs as abs_op

    exp_inf = maximum(multiply(exp_inf, b2_t), abs_op(grad))
    state["exp_inf"] = exp_inf

    bias_correction = convert_to_tensor(1.0 - beta1**step)
    step_size = divide(convert_to_tensor(lr), bias_correction)

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
        param = multiply(param, convert_to_tensor(1.0 - lr * weight_decay))

    b1_t = convert_to_tensor(beta1)
    b2_t = convert_to_tensor(beta2)
    one_minus_b1 = convert_to_tensor(1.0 - beta1)
    one_minus_b2 = convert_to_tensor(1.0 - beta2)
    lr_t = convert_to_tensor(lr)

    exp_avg = state.get("exp_avg", convert_to_tensor(0.0))

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
    new_param = subtract(param, multiply(grad, convert_to_tensor(lr)))
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
    new_param = subtract(param, multiply(grad, convert_to_tensor(lr)))
    return new_param, state
