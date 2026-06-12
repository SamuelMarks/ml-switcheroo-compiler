"""Reductions operations package."""

import ml_switcheroo_compiler.ops.reductions.basic as _basic
from ml_switcheroo_compiler.ops.base import get_op

_ = _basic

sum = get_op("Sum")()
prod = get_op("Prod")()
mean = get_op("Mean")()
variance = get_op("Variance")()
std = get_op("Std")()
max = get_op("Max")()
min = get_op("Min")()
argmax = get_op("Argmax")()
argmin = get_op("Argmin")()
all = get_op("All")()
any = get_op("Any")()

logsumexp = get_op("Logsumexp")()
count_nonzero = get_op("CountNonzero")()
norm = get_op("Norm")()
cumsum = get_op("Cumsum")()
